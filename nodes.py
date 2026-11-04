import os

import folder_paths

from .vox import write_vox, single_color_palette
from .voxelextraction import voxel_points, rotated_to_z_up, fitted, size_of
from .palettequantization import quantized, palette_of


UP_AXIS_TOOLTIP = (
    "Up axis of the incoming voxels. Texture voxels and Pixal3D structures are y-up, "
    "TRELLIS.2 structures are z-up. MagicaVoxel is z-up."
)


def next_output_path(filename_prefix):
    output_directory = folder_paths.get_output_directory()
    folder, filename, counter, _, _ = folder_paths.get_save_image_path(filename_prefix, output_directory)
    return os.path.join(folder, f"{filename}_{counter:05}_.vox")


def palette_and_indices(cell_colors, color_count, fallback_color):
    if cell_colors is None:
        return single_color_palette(fallback_color), None
    centers, indices = quantized(cell_colors, color_count)
    return palette_of(centers), indices.tolist()


def vox_voxels(cells, indices):
    positions = cells.tolist()
    if indices is None:
        return [(*position, 1) for position in positions]
    return [(*position, index + 1) for position, index in zip(positions, indices)]


class SaveVoxelsAsVox:
    CATEGORY = "vexel"
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "voxel": ("VOXEL",),
                "filename_prefix": ("STRING", {"default": "vox/trellis"}),
                "up_axis": (["y", "z"], {"tooltip": UP_AXIS_TOOLTIP}),
                "maximum_size": ("INT", {"default": 64, "min": 1, "max": 256,
                                         "tooltip": "Voxels along the longest axis, larger inputs are downsampled."}),
                "colors": ("INT", {"default": 64, "min": 1, "max": 255,
                                   "tooltip": "Palette size the base colors are quantized to."}),
                "fallback_color": ("STRING", {"default": "#b4b4b4",
                                              "tooltip": "Color for voxels without colors, such as structures."}),
                "threshold": ("FLOAT", {"default": 0.5, "min": -100.0, "max": 100.0, "step": 0.01,
                                        "tooltip": "Dense grids keep cells above this value."}),
                "batch_index": ("INT", {"default": 0, "min": 0, "max": 63}),
            }
        }

    def save(self, voxel, filename_prefix, up_axis, maximum_size, colors, fallback_color, threshold, batch_index):
        coordinates, base_colors = voxel_points(voxel, batch_index, threshold)
        oriented = rotated_to_z_up(coordinates) if up_axis == "y" else coordinates
        cells, cell_colors = fitted(oriented, base_colors, maximum_size)
        palette, indices = palette_and_indices(cell_colors, colors, fallback_color)
        path = next_output_path(filename_prefix)
        write_vox(path, size_of(cells), vox_voxels(cells, indices), palette)
        summary = f"{cells.shape[0]} voxels, size {size_of(cells)}, {len(palette)} colors, saved to {path}"
        print(f"[SaveVoxelsAsVox] {summary}")
        return {"ui": {"text": [summary]}}
