import importlib
import io
import os
import struct
import sys
import tempfile
import types
import unittest
import zipfile

import torch


PACKAGE_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PACKAGE_PARENT)


def chunk_list(content):
    chunks = []
    offset = 0
    while offset < len(content):
        identifier = content[offset:offset + 4]
        content_length, _ = struct.unpack_from("<ii", content, offset + 4)
        chunks.append((identifier, content[offset + 12:offset + 12 + content_length]))
        offset += 12 + content_length
    return chunks


def read_chunks(content):
    return dict(chunk_list(content))


def read_text(content, offset):
    length = struct.unpack_from("<i", content, offset)[0]
    return content[offset + 4:offset + 4 + length].decode(), offset + 4 + length


def read_dictionary(content, offset):
    count = struct.unpack_from("<i", content, offset)[0]
    offset += 4
    pairs = {}
    for _ in range(count):
        key, offset = read_text(content, offset)
        pairs[key], offset = read_text(content, offset)
    return pairs, offset


def read_named_transforms(chunks):
    transforms = []
    for identifier, content in chunks:
        if identifier != b"nTRN":
            continue
        attributes, offset = read_dictionary(content, 4)
        frame, _ = read_dictionary(content, offset + 16)
        if "_name" in attributes:
            transforms.append((attributes["_name"], tuple(int(part) for part in frame["_t"].split())))
    return transforms


def read_multi_model_vox(path):
    with open(path, "rb") as file:
        content = file.read()
    _, children_length = struct.unpack_from("<ii", content, 12)
    chunks = chunk_list(content[20:20 + children_length])
    sizes = [struct.unpack("<iii", data) for identifier, data in chunks if identifier == b"SIZE"]
    voxel_counts = [struct.unpack_from("<i", data)[0] for identifier, data in chunks if identifier == b"XYZI"]
    palette_data = read_chunks(content[20:20 + children_length])[b"RGBA"]
    palette = [tuple(palette_data[index * 4:index * 4 + 4]) for index in range(256)]
    return sizes, voxel_counts, read_named_transforms(chunks), palette


def read_vox(path):
    with open(path, "rb") as file:
        content = file.read()
    assert content[:4] == b"VOX "
    _, children_length = struct.unpack_from("<ii", content, 12)
    chunks = read_chunks(content[20:20 + children_length])
    size = struct.unpack("<iii", chunks[b"SIZE"])
    voxel_count = struct.unpack_from("<i", chunks[b"XYZI"])[0]
    voxels = [tuple(chunks[b"XYZI"][4 + index * 4:8 + index * 4]) for index in range(voxel_count)]
    palette = [tuple(chunks[b"RGBA"][index * 4:index * 4 + 4]) for index in range(256)]
    return size, voxels, palette


def sphere_grid(resolution, radius):
    axis = torch.arange(resolution).float() - resolution / 2
    x, y, z = torch.meshgrid(axis, axis, axis, indexing="ij")
    return ((x ** 2 + y ** 2 + z ** 2) < radius ** 2).float().unsqueeze(0)


def texture_channels(base_colors):
    material = torch.tensor([[0.0, 0.5, 1.0]]).expand(base_colors.shape[0], 3)
    return torch.cat((base_colors, material), dim=1)


class FakeVoxel:
    def __init__(self, data, voxel_colors=None, resolution=None):
        self.data = data
        self.voxel_colors = voxel_colors
        self.resolution = resolution


def fake_folder_paths(output_directory):
    module = types.ModuleType("folder_paths")
    module.get_output_directory = lambda: output_directory
    module.get_input_directory = lambda: os.path.join(output_directory, "input")
    module.get_save_image_path = lambda prefix, directory: (directory, prefix, 1, "", prefix)
    return module


class VoxTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp()
        sys.modules["folder_paths"] = fake_folder_paths(self.directory)
        self.nodes = importlib.import_module("comfyui_vox.nodes")
        self.nodes.folder_paths = sys.modules["folder_paths"]

    def save(self, voxel, **overrides):
        options = dict(filename_prefix="test", up_axis="z", maximum_size=64, colors=64,
                       fallback_color="#ff8000", threshold=0.5, batch_index=0)
        options.update(overrides)
        self.nodes.SaveVoxelsAsVox().save(voxel, **options)
        return read_vox(os.path.join(self.directory, "test_00001_.vox"))

    def test_structure_uses_fallback_color(self):
        size, voxels, palette = self.save(FakeVoxel(sphere_grid(32, 10)))
        self.assertEqual(size, (19, 19, 19))
        self.assertGreater(len(voxels), 3000)
        self.assertEqual({voxel[3] for voxel in voxels}, {1})
        self.assertEqual(palette[0], (255, 128, 0, 255))

    def test_texture_voxels_keep_their_colors(self):
        coordinates = torch.tensor([[0, 0, 0, 0], [0, 1, 0, 0], [0, 2, 0, 0]])
        base_colors = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        size, voxels, palette = self.save(FakeVoxel(coordinates, texture_channels(base_colors), 256))
        colors_by_x = {voxel[0]: palette[voxel[3] - 1] for voxel in voxels}
        self.assertEqual(size, (3, 1, 1))
        self.assertEqual(colors_by_x, {0: (255, 0, 0, 255), 1: (255, 0, 0, 255), 2: (0, 0, 255, 255)})

    def test_large_texture_is_downsampled_with_averaged_colors(self):
        coordinates = torch.tensor([[0, x, 0, 0] for x in range(1024)])
        base_colors = torch.tensor([[0.0, 0.0, 0.0] if x % 2 == 0 else [1.0, 1.0, 1.0] for x in range(1024)])
        size, voxels, palette = self.save(FakeVoxel(coordinates, texture_channels(base_colors), 1024))
        self.assertEqual(size, (64, 1, 1))
        self.assertEqual({palette[voxel[3] - 1] for voxel in voxels}, {(128, 128, 128, 255)})

    def test_many_colors_are_quantized_to_the_palette_size(self):
        generator = torch.Generator().manual_seed(1)
        coordinates = torch.tensor([[0, x, y, 0] for x in range(40) for y in range(40)])
        base_colors = torch.rand(coordinates.shape[0], 3, generator=generator)
        _, voxels, _ = self.save(FakeVoxel(coordinates, texture_channels(base_colors), 256), colors=16)
        self.assertEqual(len(voxels), 1600)
        self.assertLessEqual(max(voxel[3] for voxel in voxels), 16)

    def test_y_up_becomes_z_up(self):
        coordinates = torch.tensor([[0, 0, 0, 0], [0, 0, 3, 0]])
        size, _, _ = self.save(FakeVoxel(coordinates), up_axis="y")
        self.assertEqual(size, (1, 1, 4))

    def test_batch_index_selects_its_voxels(self):
        coordinates = torch.tensor([[0, 9, 9, 9], [1, 0, 0, 0], [1, 0, 1, 0]])
        size, voxels, _ = self.save(FakeVoxel(coordinates), batch_index=1)
        self.assertEqual(size, (1, 2, 1))
        self.assertEqual(len(voxels), 2)

    def test_saved_file_is_reported_for_download(self):
        result = self.nodes.SaveVoxelsAsVox().save(FakeVoxel(sphere_grid(8, 3)), "test", "z", 64, 64,
                                                   "#ff8000", 0.5, 0)
        self.assertEqual(result["ui"]["vox_files"], [{"filename": "test_00001_.vox", "subfolder": "", "type": "output"}])

    def test_empty_structure_is_rejected(self):
        with self.assertRaises(ValueError):
            self.save(FakeVoxel(torch.zeros(1, 8, 8, 8)))


def box_shell(side, open_top=False):
    box = torch.ones(side, side, side, dtype=torch.bool)
    box[1:-1, 1:-1, 1:-1 if not open_top else side] = False
    return box


def shell_voxel(shell, color_of=None, resolution=256):
    coordinates = shell.nonzero()
    rows = torch.cat((torch.zeros(coordinates.shape[0], 1, dtype=torch.long), coordinates), dim=1)
    if color_of is None:
        return FakeVoxel(rows, None, resolution)
    base_colors = torch.tensor([color_of(*point) for point in coordinates.tolist()])
    return FakeVoxel(rows, texture_channels(base_colors), resolution)


class FillInteriorTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp()
        sys.modules["folder_paths"] = fake_folder_paths(self.directory)
        self.nodes = importlib.import_module("comfyui_vox.nodes")
        self.nodes.folder_paths = sys.modules["folder_paths"]

    def save(self, voxel, **overrides):
        options = dict(filename_prefix="test", up_axis="z", maximum_size=64, colors=64,
                       fallback_color="#ff8000", threshold=0.5, batch_index=0)
        options.update(overrides)
        self.nodes.SaveVoxelsAsVox().save(voxel, **options)
        return read_vox(os.path.join(self.directory, "test_00001_.vox"))

    def test_the_space_inside_the_surface_is_filled(self):
        _, voxels, _ = self.save(shell_voxel(box_shell(8)))
        self.assertEqual(len(voxels), 8 ** 3)

    def test_the_surface_is_left_hollow_when_asked(self):
        _, voxels, _ = self.save(shell_voxel(box_shell(8)), fill_interior=False)
        self.assertEqual(len(voxels), 8 ** 3 - 6 ** 3)

    def test_the_inside_takes_the_color_of_the_nearest_surface(self):
        red_top = lambda x, y, z: [1.0, 0.0, 0.0] if z >= 5 else [0.0, 0.0, 1.0]
        _, voxels, palette = self.save(shell_voxel(box_shell(11), red_top))
        colors = {voxel[:3]: palette[voxel[3] - 1] for voxel in voxels}
        self.assertEqual(colors[(5, 5, 8)], (255, 0, 0, 255))
        self.assertEqual(colors[(5, 5, 2)], (0, 0, 255, 255))

    def test_an_open_surface_stays_hollow(self):
        _, voxels, _ = self.save(shell_voxel(box_shell(8, open_top=True)))
        self.assertEqual(len(voxels), int(box_shell(8, open_top=True).sum()))

    def test_grid_placement_is_filled_in_its_grid(self):
        size, voxels, _ = self.save(shell_voxel(box_shell(8), resolution=16), placement="grid", maximum_size=16)
        self.assertEqual(size, (16, 16, 16))
        self.assertEqual(len(voxels), 8 ** 3)

    def test_collection_models_are_filled(self):
        self.nodes.SaveVoxelsToVoxCollection().save(shell_voxel(box_shell(8)), "box", "vox/boxes", "z", 64, 64,
                                                    "#ff8000", 0.5, 0, 8)
        _, voxels, _ = read_vox(os.path.join(self.directory, "vox", "boxes", "box.vox"))
        self.assertEqual(len(voxels), 8 ** 3)


RED, BLUE, GREEN, CYAN, MAGENTA, YELLOW = ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 1.0, 0.0), (0.0, 1.0, 1.0),
                                           (1.0, 0.0, 1.0), (1.0, 1.0, 0.0))
BOX_LOW, BOX_HIGH, GRID_SIDE = (4, 2, 6), (12, 18, 10), 24


def hunyuan_box():
    data = torch.zeros(1, GRID_SIDE, GRID_SIDE, GRID_SIDE)
    for x in range(BOX_LOW[0], BOX_HIGH[0]):
        for y in range(BOX_LOW[1], BOX_HIGH[1]):
            for z in range(BOX_LOW[2], BOX_HIGH[2]):
                data[0, z, y, GRID_SIDE - 1 - x] = 1.0
    return FakeVoxel(data)


def halves_picture(left_color, right_color):
    image = torch.zeros(1, 40, 40, 3)
    mask = torch.ones(1, 40, 40)
    image[0, 5:35, 10:20] = torch.tensor(left_color)
    image[0, 5:35, 20:30] = torch.tensor(right_color)
    mask[0, 5:35, 10:30] = 0.0
    return image, mask


class PaintTests(unittest.TestCase):
    def setUp(self):
        sys.modules["folder_paths"] = fake_folder_paths(tempfile.mkdtemp())
        self.nodes = importlib.import_module("comfyui_vox.nodes")

    def paint(self, back=True):
        front, front_mask = halves_picture(RED, BLUE)
        back_image, back_mask = halves_picture(YELLOW, GREEN) if back else (None, None)
        painted, = self.nodes.PaintVoxelsFromViews().paint(hunyuan_box(), 0.5, front, front_mask, back_image, back_mask)
        return {tuple(cell): tuple(round(channel, 3) for channel in color)
                for cell, color in zip(painted.data.tolist(), painted.voxel_colors.tolist())}

    def test_only_the_surface_is_kept(self):
        colors = self.paint()
        self.assertNotIn((8, 10, 8), colors)
        self.assertIn((8, 10, 9), colors)

    def test_the_front_shows_the_subjects_left_on_the_right_of_the_picture(self):
        colors = self.paint()
        self.assertEqual(colors[(5, 10, 9)], RED)
        self.assertEqual(colors[(10, 10, 9)], BLUE)

    def test_the_sides_take_the_nearer_of_the_front_and_back(self):
        colors = self.paint()
        self.assertEqual(colors[(11, 10, 8)], BLUE)
        self.assertEqual(colors[(11, 10, 7)], YELLOW)
        self.assertEqual(colors[(4, 10, 8)], RED)
        self.assertEqual(colors[(4, 10, 7)], GREEN)

    def test_the_back_shows_the_subjects_left_on_the_left_of_the_picture(self):
        colors = self.paint()
        self.assertEqual(colors[(10, 10, 6)], YELLOW)
        self.assertEqual(colors[(5, 10, 6)], GREEN)

    def test_voxels_no_picture_sees_take_a_painted_color(self):
        colors = self.paint()
        painted = {RED, BLUE, YELLOW, GREEN}
        self.assertIn(colors[(8, 17, 8)], painted)
        self.assertIn(colors[(8, 2, 7)], painted)

    def test_without_a_back_picture_the_back_takes_painted_colors(self):
        colors = self.paint(back=False)
        self.assertIn(colors[(8, 10, 6)], {RED, BLUE})

    def test_painted_voxels_save_facing_the_front_of_magicavoxel(self):
        directory = tempfile.mkdtemp()
        sys.modules["folder_paths"] = fake_folder_paths(directory)
        self.nodes.folder_paths = sys.modules["folder_paths"]
        front, front_mask = halves_picture(RED, BLUE)
        painted, = self.nodes.PaintVoxelsFromViews().paint(hunyuan_box(), 0.5, front, front_mask)
        self.nodes.SaveVoxelsAsVox().save(painted, "test", "y", 64, 64, "#ff8000", 0.5, 0, "fit", False)
        size, voxels, palette = read_vox(os.path.join(directory, "test_00001_.vox"))
        self.assertEqual(size, (8, 4, 16))
        colors = {voxel[:3]: palette[voxel[3] - 1][:3] for voxel in voxels}
        self.assertEqual(colors[(1, 0, 8)], (255, 0, 0))
        self.assertEqual(colors[(6, 0, 8)], (0, 0, 255))

    def test_a_picture_without_alpha_is_refused(self):
        front, _ = halves_picture(RED, BLUE)
        with self.assertRaises(ValueError):
            self.nodes.PaintVoxelsFromViews().paint(hunyuan_box(), 0.5, front, torch.zeros(1, 64, 64))


CUBE_FACES = {
    "front": [(1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, 1)],
    "back": [(-1, -1, -1), (-1, -1, 1), (-1, 1, 1), (-1, 1, -1)],
    "up": [(-1, 1, -1), (-1, 1, 1), (1, 1, 1), (1, 1, -1)],
    "down": [(-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1)],
    "right": [(-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)],
    "left": [(-1, -1, -1), (-1, 1, -1), (1, 1, -1), (1, -1, -1)],
}
TEXTURE_U = {"front": 0.0, "right": 0.5}


def textured_cube(with_texture=True):
    from PIL import Image
    vertices, faces, uv = [], [], []
    for name, corners in CUBE_FACES.items():
        start = len(vertices)
        vertices.extend(corners)
        uv.extend([(TEXTURE_U.get(name, 1.0), 0.5)] * 4)
        faces.extend([(start, start + 1, start + 2), (start, start + 2, start + 3)])
    texture = Image.new("RGB", (3, 1))
    for column, color in enumerate(((255, 0, 0), (0, 255, 0), (0, 0, 255))):
        texture.putpixel((column, 0), color)
    material = types.SimpleNamespace(image=texture if with_texture else None)
    visual = types.SimpleNamespace(uv=uv, material=material)
    return types.SimpleNamespace(vertices=vertices, faces=faces, visual=visual)


class VoxelizeTexturedMeshTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp()
        sys.modules["folder_paths"] = fake_folder_paths(self.directory)
        self.nodes = importlib.import_module("comfyui_vox.nodes")
        self.nodes.folder_paths = sys.modules["folder_paths"]

    def voxelize(self, mesh=None):
        voxels, = self.nodes.VoxelizeTexturedMesh().voxelize(mesh or textured_cube(), 8, 8)
        return voxels, {tuple(cell): tuple(round(channel, 3) for channel in color)
                        for cell, color in zip(voxels.data.tolist(), voxels.voxel_colors.tolist())}

    def test_every_surface_voxel_is_sampled(self):
        voxels, colors = self.voxelize()
        self.assertEqual(len(colors), 8 ** 3 - 6 ** 3)
        self.assertEqual(voxels.resolution, 8)

    def test_voxels_take_the_texture_of_the_faces_they_cover(self):
        _, colors = self.voxelize()
        self.assertEqual(colors[(4, 4, 7)], (1.0, 0.0, 0.0))
        self.assertEqual(colors[(0, 4, 4)], (0.0, 1.0, 0.0))
        self.assertEqual(colors[(7, 4, 4)], (0.0, 0.0, 1.0))
        self.assertEqual(colors[(4, 7, 4)], (0.0, 0.0, 1.0))

    def test_a_mesh_without_texture_is_refused(self):
        with self.assertRaises(ValueError):
            self.voxelize(textured_cube(with_texture=False))

    def test_voxelized_meshes_save_facing_the_front_of_magicavoxel(self):
        voxels, _ = self.voxelize()
        self.nodes.SaveVoxelsAsVox().save(voxels, "test", "y", 64, 64, "#ff8000", 0.5, 0, "fit", True)
        size, saved, palette = read_vox(os.path.join(self.directory, "test_00001_.vox"))
        colors = {voxel[:3]: palette[voxel[3] - 1][:3] for voxel in saved}
        self.assertEqual(size, (8, 8, 8))
        self.assertEqual(len(saved), 8 ** 3)
        self.assertEqual(colors[(4, 0, 4)], (255, 0, 0))
        self.assertEqual(colors[(0, 4, 4)], (0, 255, 0))


class FramingBackgroundTests(unittest.TestCase):
    def setUp(self):
        sys.modules["folder_paths"] = fake_folder_paths(tempfile.mkdtemp())
        self.nodes = importlib.import_module("comfyui_vox.nodes")

    def test_the_background_fills_around_the_subject(self):
        front, front_mask = halves_picture(RED, BLUE)
        side, side_mask = halves_picture(CYAN, MAGENTA)
        framed, _, _, _ = self.nodes.FrameFrontAndSideViews().frame(front, front_mask, side, side_mask, "left",
                                                                    background=0.8)
        self.assertAlmostEqual(framed[0, 0, 0, 0].item(), 0.8, places=5)
        middle = framed.shape[1] // 2
        self.assertEqual(framed[0, middle, middle // 2].tolist(), [1.0, 0.0, 0.0])

    def test_the_background_is_black_by_default(self):
        front, front_mask = halves_picture(RED, BLUE)
        side, side_mask = halves_picture(CYAN, MAGENTA)
        framed, _, _, _ = self.nodes.FrameFrontAndSideViews().frame(front, front_mask, side, side_mask, "left")
        self.assertEqual(framed[0, 0, 0].tolist(), [0.0, 0.0, 0.0])


def colored_line(length, color):
    coordinates = torch.tensor([[0, x, 0, 0] for x in range(length)])
    base_colors = torch.tensor([color] * length)
    return FakeVoxel(coordinates, texture_channels(base_colors), 256)


class CollectionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp()
        sys.modules["folder_paths"] = fake_folder_paths(self.directory)
        self.nodes = importlib.import_module("comfyui_vox.nodes")
        self.nodes.folder_paths = sys.modules["folder_paths"]

    def add(self, voxel, name, **overrides):
        options = dict(collection="vox/characters", up_axis="z", maximum_size=64, colors=64,
                       fallback_color="#ff8000", threshold=0.5, batch_index=0, spacing=8)
        options.update(overrides)
        return self.nodes.SaveVoxelsToVoxCollection().save(voxel, name, **options)

    def collection_file(self):
        return os.path.join(self.directory, "vox", "characters.vox")

    def test_models_are_kept_apart_and_merged_in_numeric_order(self):
        self.add(colored_line(4, [1.0, 0.0, 0.0]), "10")
        self.add(colored_line(2, [0.0, 0.0, 1.0]), "2")
        sizes, voxel_counts, transforms, palette = read_multi_model_vox(self.collection_file())
        models_folder = os.path.join(self.directory, "vox", "characters")
        self.assertEqual(sorted(os.listdir(models_folder)), ["10.vox", "2.vox"])
        self.assertEqual(sizes, [(2, 1, 1), (4, 1, 1)])
        self.assertEqual(voxel_counts, [2, 4])
        self.assertEqual([name for name, _ in transforms], ["2", "10"])
        self.assertEqual(set(palette[:2]), {(255, 0, 0, 255), (0, 0, 255, 255)})

    def test_models_stand_side_by_side_on_the_ground(self):
        self.add(colored_line(4, [1.0, 0.0, 0.0]), "1")
        self.add(colored_line(4, [1.0, 0.0, 0.0]), "2")
        _, _, transforms, _ = read_multi_model_vox(self.collection_file())
        self.assertEqual([translation for _, translation in transforms], [(2, 0, 0), (14, 0, 0)])

    def test_a_model_with_the_same_name_is_replaced(self):
        self.add(colored_line(4, [1.0, 0.0, 0.0]), "7")
        self.add(colored_line(3, [1.0, 0.0, 0.0]), "7")
        sizes, _, _, _ = read_multi_model_vox(self.collection_file())
        self.assertEqual(sizes, [(3, 1, 1)])

    def test_unnamed_models_are_numbered(self):
        self.add(colored_line(1, [1.0, 0.0, 0.0]), "")
        self.add(colored_line(1, [1.0, 0.0, 0.0]), "")
        _, _, transforms, _ = read_multi_model_vox(self.collection_file())
        self.assertEqual([name for name, _ in transforms], ["001", "002"])

    def test_shared_palette_is_quantized_to_the_palette_size(self):
        generator = torch.Generator().manual_seed(2)
        for name in ("1", "2", "3"):
            coordinates = torch.tensor([[0, x, y, 0] for x in range(20) for y in range(20)])
            base_colors = torch.rand(coordinates.shape[0], 3, generator=generator)
            self.add(FakeVoxel(coordinates, texture_channels(base_colors), 256), name, colors=32)
        _, voxel_counts, _, palette = read_multi_model_vox(self.collection_file())
        self.assertEqual(voxel_counts, [400, 400, 400])
        self.assertEqual(palette[32:], [(0, 0, 0, 255)] * 224)

    def test_collection_is_reported_for_download(self):
        result = self.add(colored_line(1, [1.0, 0.0, 0.0]), "1")
        self.assertEqual(result["ui"]["vox_files"], [{"filename": "characters.vox", "subfolder": "vox", "type": "output"}])

    def test_collection_outside_the_output_directory_is_refused(self):
        with self.assertRaises(ValueError):
            self.add(colored_line(1, [1.0, 0.0, 0.0]), "1", collection="../escaped")


def png_bytes(alpha, mode="RGBA"):
    from PIL import Image
    output = io.BytesIO()
    Image.new(mode, (4, 3), (255, 0, 0, alpha)[:len(mode)]).save(output, format="PNG")
    return output.getvalue()


def smudged_png_bytes():
    """Red subject on the left half, a fuchsia smudge at alpha 1/255 on the right."""
    from PIL import Image
    picture = Image.new("RGBA", (4, 2), (255, 0, 255, 1))
    for x in range(2):
        for y in range(2):
            picture.putpixel((x, y), (255, 0, 0, 255))
    output = io.BytesIO()
    picture.save(output, format="PNG")
    return output.getvalue()


class LoaderTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp()
        sys.modules["folder_paths"] = fake_folder_paths(self.directory)
        self.nodes = importlib.import_module("comfyui_vox.nodes")
        self.nodes.folder_paths = sys.modules["folder_paths"]
        os.makedirs(os.path.join(self.directory, "input", "sets"))
        self.write_zip("sets/heroes.zip", {
            "heroes/hero-10.png": png_bytes(0),
            "heroes/hero-2.png": png_bytes(255),
            "hero-1.png": png_bytes(255),
            "__MACOSX/heroes/._hero-2.png": b"resource fork",
            "heroes/cover.png": png_bytes(255),
            "heroes/notes.txt": b"",
        })

    def write_zip(self, name, entries):
        with zipfile.ZipFile(os.path.join(self.directory, "input", name), "w") as archive:
            for entry, content in entries.items():
                archive.writestr(entry, content)

    def test_zips_are_offered_from_the_input_directory(self):
        self.write_zip("loose.zip", {})
        offered = self.nodes.LoadNumberedImage.INPUT_TYPES()["required"]["zip"][0]
        self.assertEqual(offered, ["loose.zip", "sets/heroes.zip"])

    def test_pictures_load_in_order_of_their_numbers(self):
        names = [self.load("sets/heroes.zip", index)[2] for index in range(3)]
        self.assertEqual(names, ["1", "2", "10"])

    def load(self, archive, index, whole_zip=True):
        return self.nodes.LoadNumberedImage().load(archive, index, whole_zip)["result"]

    def test_picture_becomes_image_foreground_and_count(self):
        image, foreground, name, count, remove_background = self.load("sets/heroes.zip", 1)
        self.assertEqual(tuple(image.shape), (1, 3, 4, 3))
        self.assertEqual(tuple(foreground.shape), (1, 3, 4))
        self.assertEqual(float(foreground.min()), 1.0)
        self.assertEqual(count, 3)
        self.assertFalse(remove_background)

    def test_transparent_picture_is_on_black_without_foreground(self):
        image, foreground, _, _, _ = self.load("sets/heroes.zip", 2)
        self.assertEqual(float(image.max()), 0.0)
        self.assertEqual(float(foreground.max()), 0.0)

    def test_faint_smudges_are_cleared_from_the_foreground_and_image(self):
        self.write_zip("smudged.zip", {"girl-1.png": smudged_png_bytes()})
        image, foreground, _, _, _ = self.load("smudged.zip", 0)
        self.assertEqual(foreground[0].tolist(), [[1.0, 1.0, 0.0, 0.0]] * 2)
        self.assertEqual(image[0, :, 2:].abs().sum().item(), 0.0)
        self.assertEqual(image[0, 0, 0].tolist(), [1.0, 0.0, 0.0])

    def test_picture_without_alpha_needs_its_background_removed(self):
        self.write_zip("opaque.zip", {"girl-1.png": png_bytes(255, mode="RGB")})
        _, foreground, _, _, remove_background = self.load("opaque.zip", 0)
        self.assertTrue(remove_background)
        self.assertEqual(float(foreground.min()), 1.0)

    def test_progress_is_reported_for_queueing_the_next_picture(self):
        output = self.nodes.LoadNumberedImage().load("sets/heroes.zip", 1, True)
        self.assertEqual(output["ui"]["picture"], [{"index": 1, "count": 3, "whole_zip": True}])

    def test_index_past_the_last_picture_ends_the_batch(self):
        with self.assertRaises(ValueError):
            self.load("sets/heroes.zip", 3)

    def test_pictures_sharing_a_number_are_refused(self):
        self.write_zip("twins.zip", {"a-1.png": png_bytes(255), "b-01.png": png_bytes(255)})
        with self.assertRaises(ValueError):
            self.load("twins.zip", 0)

    def test_zip_outside_the_input_directory_is_refused(self):
        with self.assertRaises(ValueError):
            self.load("../escaped.zip", 0)


class GridPlacementTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp()
        sys.modules["folder_paths"] = fake_folder_paths(self.directory)
        self.nodes = importlib.import_module("comfyui_vox.nodes")
        self.nodes.folder_paths = sys.modules["folder_paths"]

    def save(self, voxel, **overrides):
        options = dict(filename_prefix="test", up_axis="y", maximum_size=64, colors=64,
                       fallback_color="#ff8000", threshold=0.5, batch_index=0, placement="grid")
        options.update(overrides)
        self.nodes.SaveVoxelsAsVox().save(voxel, **options)
        path = os.path.join(self.directory, "test_00001_.vox")
        result = read_vox(path)
        os.remove(path)
        return result

    def test_models_keep_their_place_in_the_grid(self):
        size, voxels, _ = self.save(FakeVoxel(torch.tensor([[0, 512, 1000, 256]]), None, 1024))
        self.assertEqual(size, (64, 64, 64))
        self.assertEqual([voxel[:3] for voxel in voxels], [(32, 47, 62)])

    def test_differently_shaped_models_share_one_size(self):
        small, _, _ = self.save(FakeVoxel(torch.tensor([[0, 10, 10, 10]]), None, 256))
        large, _, _ = self.save(FakeVoxel(torch.tensor([[0, 10, 10, 10], [0, 200, 200, 200]]), None, 256))
        self.assertEqual(small, large)

    def test_dense_structures_scale_from_their_grid(self):
        grid = torch.zeros(1, 32, 32, 32)
        grid[0, 31, 0, 0] = 1
        size, voxels, _ = self.save(FakeVoxel(grid), up_axis="z")
        self.assertEqual(size, (32, 32, 32))
        self.assertEqual(voxels[0][:3], (31, 0, 0))


def view_png_bytes(left, top, width, height, size=(10, 8)):
    from PIL import Image
    picture = Image.new("RGBA", size, (0, 0, 0, 0))
    for x in range(left, left + width):
        for y in range(top, top + height):
            picture.putpixel((x, y), (255, 255, 255, 255))
    output = io.BytesIO()
    picture.save(output, format="PNG")
    return output.getvalue()


class ViewLoaderTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp()
        sys.modules["folder_paths"] = fake_folder_paths(self.directory)
        self.nodes = importlib.import_module("comfyui_vox.nodes")
        self.nodes.folder_paths = sys.modules["folder_paths"]
        os.makedirs(os.path.join(self.directory, "input"))

    def write_zip(self, name, entries):
        with zipfile.ZipFile(os.path.join(self.directory, "input", name), "w") as archive:
            for entry, content in entries.items():
                archive.writestr(entry, content)

    def load(self, archive, index):
        return self.nodes.LoadNumberedViews().load(archive, index, True)["result"]

    def write_run(self):
        self.write_zip("run.zip", {
            "girl-run-left-2.png": view_png_bytes(4, 2, 2, 4),
            "girl-run-left-1.png": view_png_bytes(2, 2, 2, 4),
            "front/girl-run-1.png": view_png_bytes(1, 0, 3, 6),
            "front/girl-run-2.png": view_png_bytes(1, 1, 3, 6),
        })

    def test_frames_come_in_order_with_their_views(self):
        self.write_run()
        front, left, back, right, name, count = self.load("run.zip", 1)
        self.assertEqual((name, count), ("2", 2))
        self.assertIsNotNone(front)
        self.assertIsNotNone(left)
        self.assertIsNone(back)
        self.assertIsNone(right)

    def test_every_view_and_frame_share_one_square(self):
        self.write_run()
        first = self.load("run.zip", 0)
        second = self.load("run.zip", 1)
        shapes = {tuple(image.shape) for image in (*first[:2], *second[:2])}
        self.assertEqual(shapes, {(1, 8, 8, 3)})

    def test_the_animation_keeps_its_place_in_the_square(self):
        self.write_run()
        first_left = self.load("run.zip", 0)[1][0, :, :, 0]
        second_left = self.load("run.zip", 1)[1][0, :, :, 0]
        first_columns = first_left.amax(dim=0).nonzero().flatten().tolist()
        second_columns = second_left.amax(dim=0).nonzero().flatten().tolist()
        self.assertEqual(second_columns[0] - first_columns[0], 2)

    def test_frames_missing_a_view_are_refused(self):
        self.write_zip("gap.zip", {
            "girl-left-1.png": view_png_bytes(2, 2, 2, 4),
            "girl-left-2.png": view_png_bytes(2, 2, 2, 4),
            "girl-front-1.png": view_png_bytes(2, 2, 2, 4),
        })
        with self.assertRaises(ValueError):
            self.load("gap.zip", 0)

    def test_pictures_without_a_view_are_refused(self):
        self.write_zip("plain.zip", {"girl-1.png": view_png_bytes(2, 2, 2, 4)})
        with self.assertRaises(ValueError):
            self.load("plain.zip", 0)


def loaded_image(width, height, left, top, right, bottom, faint_left=None):
    image = torch.zeros((1, height, width, 3))
    alpha = torch.zeros((height, width))
    image[0, top:bottom, left:right] = torch.tensor([0.8, 0.4, 0.2])
    alpha[top:bottom, left:right] = 1.0
    if faint_left is not None:
        image[0, 0, faint_left] = torch.tensor([1.0, 0.0, 1.0])
        alpha[0, faint_left] = 1 / 255
    return image, (1.0 - alpha).unsqueeze(0)


class FrontAndSideTests(unittest.TestCase):
    def setUp(self):
        sys.modules["folder_paths"] = fake_folder_paths(tempfile.mkdtemp())
        self.nodes = importlib.import_module("comfyui_vox.nodes")

    def frame(self, side_faces, front=None, side=None, back=(None, None)):
        front_image, front_mask = front or loaded_image(20, 20, 6, 2, 14, 18)
        side_image, side_mask = side or loaded_image(20, 20, 2, 4, 12, 18)
        return self.nodes.FrameFrontAndSideViews().frame(front_image, front_mask, side_image, side_mask, side_faces,
                                                         *back)

    def test_the_side_goes_to_the_view_it_faces(self):
        front, left, right, _ = self.frame("left")
        self.assertIsNotNone(left)
        self.assertIsNone(right)
        front, left, right, _ = self.frame("right")
        self.assertIsNone(left)
        self.assertIsNotNone(right)

    def test_both_views_share_one_square_at_one_height(self):
        front, left, _, _ = self.frame("left")
        self.assertEqual(tuple(front.shape), (1, 18, 18, 3))
        self.assertEqual(tuple(left.shape), (1, 18, 18, 3))
        front_rows = torch.nonzero(front[0].sum(dim=(1, 2)))[:, 0]
        side_rows = torch.nonzero(left[0].sum(dim=(1, 2)))[:, 0]
        self.assertEqual(int(front_rows.max()), int(side_rows.max()))

    def test_each_view_is_centered_on_its_subject(self):
        front, left, _, _ = self.frame("left")
        for view, width in ((front, 8), (left, 10)):
            columns = torch.nonzero(view[0].sum(dim=(0, 2)))[:, 0]
            self.assertEqual(int(columns.min()) + int(columns.max()) + 1, 18)
            self.assertEqual(int(columns.max() - columns.min()) + 1, width)

    def test_faint_smudges_are_left_out(self):
        front, _, _, _ = self.frame("left", front=loaded_image(20, 20, 6, 2, 14, 18, faint_left=0))
        self.assertEqual(tuple(front.shape), (1, 18, 18, 3))
        self.assertAlmostEqual(float(front[0, ..., 2].max()), 0.2, places=5)

    def test_the_back_is_left_out_unless_connected(self):
        self.assertIsNone(self.frame("left")[3])

    def test_the_back_shares_the_square_and_is_centered(self):
        back = loaded_image(20, 20, 9, 1, 15, 18)
        front, _, _, framed_back = self.frame("left", back=back)
        self.assertEqual(tuple(front.shape), (1, 19, 19, 3))
        self.assertEqual(tuple(framed_back.shape), (1, 19, 19, 3))
        columns = torch.nonzero(framed_back[0].sum(dim=(0, 2)))[:, 0]
        self.assertLessEqual(abs(int(columns.min()) + int(columns.max()) + 1 - 19), 1)

    def test_a_back_without_alpha_is_refused(self):
        image, _ = loaded_image(20, 20, 9, 1, 15, 18)
        with self.assertRaisesRegex(ValueError, "back picture has no transparency"):
            self.frame("left", back=(image, None))

    def test_pictures_without_alpha_are_refused(self):
        image, _ = loaded_image(20, 20, 6, 2, 14, 18)
        with self.assertRaisesRegex(ValueError, "front picture has no transparency"):
            self.frame("left", front=(image, torch.zeros((1, 64, 64))))
        with self.assertRaisesRegex(ValueError, "side picture has no transparency"):
            self.frame("left", side=(image, torch.zeros((1, 20, 20))))


if __name__ == "__main__":
    unittest.main()
