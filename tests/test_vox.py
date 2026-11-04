import importlib
import os
import struct
import sys
import tempfile
import types
import unittest

import torch


PACKAGE_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PACKAGE_PARENT)


def read_chunks(content):
    chunks = {}
    offset = 0
    while offset < len(content):
        identifier = content[offset:offset + 4]
        content_length, _ = struct.unpack_from("<ii", content, offset + 4)
        chunks[identifier] = content[offset + 12:offset + 12 + content_length]
        offset += 12 + content_length
    return chunks


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

    def test_empty_structure_is_rejected(self):
        with self.assertRaises(ValueError):
            self.save(FakeVoxel(torch.zeros(1, 8, 8, 8)))


if __name__ == "__main__":
    unittest.main()
