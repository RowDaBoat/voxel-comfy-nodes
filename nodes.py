import os

import folder_paths

from .imageloading import image_and_mask, image_paths_in
from .naming import folder_inside, safe_name, stem_of
from .palettequantization import quantized, palette_of
from .vox import write_vox, single_color_palette
from .voxcollection import write_collection
from .voxelextraction import voxel_points, rotated_to_z_up, fitted, size_of


UP_AXIS_TOOLTIP = (
    "Up axis of the incoming voxels. Texture voxels and Pixal3D structures are y-up, "
    "TRELLIS.2 structures are z-up. MagicaVoxel is z-up."
)


MAXIMUM_INDEX = 99999


def voxel_options():
    return {
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


def voxel_cells(voxel, up_axis, maximum_size, threshold, batch_index):
    coordinates, base_colors = voxel_points(voxel, batch_index, threshold)
    oriented = rotated_to_z_up(coordinates) if up_axis == "y" else coordinates
    return fitted(oriented, base_colors, maximum_size)


def saved_file_of(path):
    """What ComfyUI's `/view` needs to serve a file of the output directory."""
    output_directory = os.path.realpath(folder_paths.get_output_directory())
    subfolder = os.path.relpath(os.path.dirname(os.path.realpath(path)), output_directory)
    subfolder = "" if subfolder == "." else subfolder.replace(os.sep, "/")
    return {"filename": os.path.basename(path), "subfolder": subfolder, "type": "output"}


def next_output_file(filename_prefix):
    output_directory = folder_paths.get_output_directory()
    folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(filename_prefix, output_directory)
    vox_filename = f"{filename}_{counter:05}_.vox"
    return os.path.join(folder, vox_filename), {"filename": vox_filename, "subfolder": subfolder, "type": "output"}


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
                **voxel_options(),
            }
        }

    def save(self, voxel, filename_prefix, up_axis, maximum_size, colors, fallback_color, threshold, batch_index):
        cells, cell_colors = voxel_cells(voxel, up_axis, maximum_size, threshold, batch_index)
        palette, indices = palette_and_indices(cell_colors, colors, fallback_color)
        path, vox_file = next_output_file(filename_prefix)
        write_vox(path, size_of(cells), vox_voxels(cells, indices), palette)
        summary = f"{cells.shape[0]} voxels, size {size_of(cells)}, {len(palette)} colors, saved to {path}"
        print(f"[SaveVoxelsAsVox] {summary}")
        return {"ui": {"text": [summary], "vox_files": [vox_file]}}


def next_model_name(models_folder):
    number = len(os.listdir(models_folder)) + 1
    while os.path.exists(os.path.join(models_folder, f"{number:03}.vox")):
        number += 1
    return f"{number:03}"


def collection_paths(collection):
    output_directory = folder_paths.get_output_directory()
    models_folder = folder_inside(output_directory, collection)
    if models_folder == os.path.realpath(output_directory):
        raise ValueError("the collection needs a name, such as vox/characters")
    return models_folder, models_folder + ".vox"


class SaveVoxelsToVoxCollection:
    """Adds one model per run to a collection folder and rewrites the collection's
    multi-model .vox from every model in it, with one palette shared by all."""

    CATEGORY = "vexel"
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "save"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "voxel": ("VOXEL",),
                "name": ("STRING", {"default": "", "tooltip": (
                    "Model name, wire Load Numbered Image's name here. A model with the same name is "
                    "replaced, an empty name numbers the model after the ones in the collection.")}),
                "collection": ("STRING", {"default": "vox/collection", "tooltip": (
                    "Saves each model to <collection>/<name>.vox and all of them to <collection>.vox, "
                    "inside ComfyUI's output directory.")}),
                **voxel_options(),
                "spacing": ("INT", {"default": 8, "min": 0, "max": 256,
                                    "tooltip": "Empty voxels between models in the multi-model file."}),
            }
        }

    def save(self, voxel, name, collection, up_axis, maximum_size, colors, fallback_color, threshold,
             batch_index, spacing):
        models_folder, collection_path = collection_paths(collection)
        os.makedirs(models_folder, exist_ok=True)
        model_name = safe_name(name, next_model_name(models_folder))
        cells, cell_colors = voxel_cells(voxel, up_axis, maximum_size, threshold, batch_index)
        palette, indices = palette_and_indices(cell_colors, colors, fallback_color)
        write_vox(os.path.join(models_folder, f"{model_name}.vox"), size_of(cells), vox_voxels(cells, indices), palette)
        models, shared_palette = write_collection(models_folder, collection_path, colors, spacing)
        summary = (f"{model_name}: {cells.shape[0]} voxels, size {size_of(cells)}. "
                   f"{collection_path} holds {len(models)} models sharing {len(shared_palette)} colors")
        print(f"[SaveVoxelsToVoxCollection] {summary}")
        return {"ui": {"text": [summary], "vox_files": [saved_file_of(collection_path)]}}


def picture_paths(folder):
    return image_paths_in(folder_inside(folder_paths.get_input_directory(), folder))


class LoadNumberedImage:
    """One picture of a folder per run, in numeric order: queue as many runs as there are
    pictures with `index` on increment."""

    CATEGORY = "vexel"
    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "INT")
    RETURN_NAMES = ("image", "mask", "name", "count")
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder": ("STRING", {"default": "characters",
                                      "tooltip": "Folder inside ComfyUI's input directory."}),
                "index": ("INT", {"default": 0, "min": 0, "max": MAXIMUM_INDEX, "control_after_generate": True,
                                  "tooltip": "Which picture, counting from 0 in numeric order of the file names."}),
            }
        }

    @classmethod
    def IS_CHANGED(cls, folder, index):
        input_folder = os.path.join(folder_paths.get_input_directory(), folder)
        if not os.path.isdir(input_folder):
            return folder
        return [(path, os.path.getmtime(path)) for path in picture_paths(folder)]

    def load(self, folder, index):
        paths = picture_paths(folder)
        if index >= len(paths):
            raise ValueError(f"index {index} is past the {len(paths)} pictures in {folder}, the batch is done")
        image, mask = image_and_mask(paths[index])
        print(f"[LoadNumberedImage] {index + 1}/{len(paths)} {paths[index]}")
        return image, mask, stem_of(paths[index]), len(paths)
