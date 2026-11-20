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

    def frame(self, side_faces, front=None, side=None):
        front_image, front_mask = front or loaded_image(20, 20, 6, 2, 14, 18)
        side_image, side_mask = side or loaded_image(20, 20, 2, 4, 12, 18)
        return self.nodes.FrameFrontAndSideViews().frame(front_image, front_mask, side_image, side_mask, side_faces)

    def test_the_side_goes_to_the_view_it_faces(self):
        front, left, right = self.frame("left")
        self.assertIsNotNone(left)
        self.assertIsNone(right)
        front, left, right = self.frame("right")
        self.assertIsNone(left)
        self.assertIsNotNone(right)

    def test_both_views_share_one_square_at_one_height(self):
        front, left, _ = self.frame("left")
        self.assertEqual(tuple(front.shape), (1, 18, 18, 3))
        self.assertEqual(tuple(left.shape), (1, 18, 18, 3))
        front_rows = torch.nonzero(front[0].sum(dim=(1, 2)))[:, 0]
        side_rows = torch.nonzero(left[0].sum(dim=(1, 2)))[:, 0]
        self.assertEqual(int(front_rows.max()), int(side_rows.max()))

    def test_each_view_is_centered_on_its_subject(self):
        front, left, _ = self.frame("left")
        for view, width in ((front, 8), (left, 10)):
            columns = torch.nonzero(view[0].sum(dim=(0, 2)))[:, 0]
            self.assertEqual(int(columns.min()) + int(columns.max()) + 1, 18)
            self.assertEqual(int(columns.max() - columns.min()) + 1, width)

    def test_faint_smudges_are_left_out(self):
        front, _, _ = self.frame("left", front=loaded_image(20, 20, 6, 2, 14, 18, faint_left=0))
        self.assertEqual(tuple(front.shape), (1, 18, 18, 3))
        self.assertAlmostEqual(float(front[0, ..., 2].max()), 0.2, places=5)

    def test_pictures_without_alpha_are_refused(self):
        image, _ = loaded_image(20, 20, 6, 2, 14, 18)
        with self.assertRaisesRegex(ValueError, "front picture has no transparency"):
            self.frame("left", front=(image, torch.zeros((1, 64, 64))))
        with self.assertRaisesRegex(ValueError, "side picture has no transparency"):
            self.frame("left", side=(image, torch.zeros((1, 20, 20))))


if __name__ == "__main__":
    unittest.main()
