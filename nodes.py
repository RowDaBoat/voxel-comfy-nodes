import os

import folder_paths

from .imageloading import archives_in, numbered_picture
from .viewloading import VIEW_NAMES, framed_views, numbered_views, premultiplied_image
from .naming import folder_inside, safe_name
from .palettequantization import quantized, palette_of
from .vox import write_vox, single_color_palette
from .voxcollection import write_collection
from .voxelpainting import Picture, hunyuan_occupancy, painted_voxels, views_of
from .meshvoxelization import textured_mesh_voxels
from .voxelextraction import (voxel_points, rotated_to_z_up, rotated_to_z_up_in_grid, fitted, size_of,
                              grid_resolution_of, placed_in_grid, grid_size_of, with_filled_interior)


UP_AXIS_TOOLTIP = (
    "Up axis of the incoming voxels. Texture voxels and Pixal3D structures are y-up, "
    "TRELLIS.2 structures are z-up. MagicaVoxel is z-up."
)


MAXIMUM_INDEX = 99999
PLACEMENT_TOOLTIP = (
    "fit: the model is cropped to its own bounds. grid: the model keeps its place in TRELLIS's grid "
    "and every model is the same size, so frames of an animation line up."
)
FILL_INTERIOR_TOOLTIP = (
    "TRELLIS only makes the surface. Fills the space enclosed by it with the color of the nearest surface voxel."
)


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


def placement_option():
    return {"placement": (["fit", "grid"], {"tooltip": PLACEMENT_TOOLTIP}),
            "fill_interior": ("BOOLEAN", {"default": True, "tooltip": FILL_INTERIOR_TOOLTIP})}


def fitted_cells(voxel, up_axis, maximum_size, threshold, batch_index):
    coordinates, base_colors = voxel_points(voxel, batch_index, threshold)
    oriented = rotated_to_z_up(coordinates) if up_axis == "y" else coordinates
    cells, cell_colors = fitted(oriented, base_colors, maximum_size)
    return cells, cell_colors, size_of(cells)


def grid_cells(voxel, up_axis, maximum_size, threshold, batch_index):
    coordinates, base_colors = voxel_points(voxel, batch_index, threshold)
    resolution = grid_resolution_of(voxel)
    oriented = rotated_to_z_up_in_grid(coordinates, resolution) if up_axis == "y" else coordinates
    cells, cell_colors = placed_in_grid(oriented, base_colors, maximum_size, resolution)
    return cells, cell_colors, grid_size_of(maximum_size, resolution)


def voxel_cells(voxel, up_axis, maximum_size, threshold, batch_index, placement="fit"):
    """Cells, their averaged colors, and the model size."""
    cells_of = grid_cells if placement == "grid" else fitted_cells
    return cells_of(voxel, up_axis, maximum_size, threshold, batch_index)


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
    return palette_of(centers), indices


def saved_voxels(cells, indices, size, fill_interior):
    if fill_interior:
        cells, indices = with_filled_interior(cells, indices, size)
    return cells, vox_voxels(cells, indices)


def vox_voxels(cells, indices):
    positions = cells.tolist()
    if indices is None:
        return [(*position, 1) for position in positions]
    return [(*position, index + 1) for position, index in zip(positions, indices.tolist())]


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
            },
            "optional": placement_option(),
        }

    def save(self, voxel, filename_prefix, up_axis, maximum_size, colors, fallback_color, threshold, batch_index,
             placement="fit", fill_interior=True):
        cells, cell_colors, size = voxel_cells(voxel, up_axis, maximum_size, threshold, batch_index, placement)
        palette, indices = palette_and_indices(cell_colors, colors, fallback_color)
        cells, voxels = saved_voxels(cells, indices, size, fill_interior)
        path, vox_file = next_output_file(filename_prefix)
        write_vox(path, size, voxels, palette)
        summary = f"{cells.shape[0]} voxels, size {size}, {len(palette)} colors, saved to {path}"
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
            },
            "optional": placement_option(),
        }

    def save(self, voxel, name, collection, up_axis, maximum_size, colors, fallback_color, threshold,
             batch_index, spacing, placement="fit", fill_interior=True):
        models_folder, collection_path = collection_paths(collection)
        os.makedirs(models_folder, exist_ok=True)
        model_name = safe_name(name, next_model_name(models_folder))
        cells, cell_colors, size = voxel_cells(voxel, up_axis, maximum_size, threshold, batch_index, placement)
        palette, indices = palette_and_indices(cell_colors, colors, fallback_color)
        cells, voxels = saved_voxels(cells, indices, size, fill_interior)
        write_vox(os.path.join(models_folder, f"{model_name}.vox"), size, voxels, palette)
        models, shared_palette = write_collection(models_folder, collection_path, colors, spacing)
        summary = (f"{model_name}: {cells.shape[0]} voxels, size {size}. "
                   f"{collection_path} holds {len(models)} models sharing {len(shared_palette)} colors")
        print(f"[SaveVoxelsToVoxCollection] {summary}")
        return {"ui": {"text": [summary], "vox_files": [saved_file_of(collection_path)]}}


def archive_path_of(archive):
    return folder_inside(folder_paths.get_input_directory(), archive)


class LoadNumberedImage:
    """One picture of a zip per run, in the order of the numbers their names end in
    (`hero-1.png`, `hero-2.png`, ... `hero-10.png`). With `whole_zip` the frontend queues
    the next picture after each run until the zip is done."""

    CATEGORY = "vexel"
    RETURN_TYPES = ("IMAGE", "MASK", "STRING", "INT", "BOOLEAN")
    RETURN_NAMES = ("image", "foreground", "name", "count", "remove_background")
    OUTPUT_TOOLTIPS = (
        "The picture, on black where it is transparent.",
        "1 on the subject, like Remove Background's mask: the picture's alpha, or all of it without alpha.",
        "The number the picture's name ends in.",
        "How many numbered pictures the zip holds.",
        "True when the picture has no alpha, so its background still has to be removed.",
    )
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "zip": (archives_in(folder_paths.get_input_directory()), {
                    "tooltip": "Zip of pictures named <anything>-<number>.png, in ComfyUI's input directory."}),
                "index": ("INT", {"default": 0, "min": 0, "max": MAXIMUM_INDEX, "control_after_generate": True,
                                  "tooltip": "Which picture, counting from 0 in the order of their numbers."}),
                "whole_zip": ("BOOLEAN", {"default": True, "tooltip": (
                    "Queue the next picture after each run, until the zip is done. "
                    "Off, each Run does one picture.")}),
            }
        }

    @classmethod
    def IS_CHANGED(cls, zip, index, whole_zip):
        archive_path = archive_path_of(zip)
        return os.path.getmtime(archive_path) if os.path.isfile(archive_path) else zip

    def load(self, zip, index, whole_zip):
        image, foreground, remove_background, number, count, entry = numbered_picture(archive_path_of(zip), index)
        print(f"[LoadNumberedImage] {index + 1}/{count} {zip}: {entry}")
        picture = {"index": index, "count": count, "whole_zip": whole_zip}
        return {"ui": {"picture": [picture]}, "result": (image, foreground, number, count, remove_background)}


class LoadNumberedViews:
    """One frame of a zip per run, every view of it framed for Pixal3D Multi-View Conditioning.
    Pictures are named `<anything>-<number>.png` with `front`, `left`, `back` or `right` in their
    name or folder, and every frame has the same views."""

    CATEGORY = "vexel"
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE", "STRING", "INT")
    RETURN_NAMES = (*VIEW_NAMES, "name", "count")
    OUTPUT_TOOLTIPS = (
        *(f"The {view} view on black, framed like every other view and frame; nothing when the zip has none."
          for view in VIEW_NAMES),
        "The frame's number.",
        "How many frames the zip holds.",
    )
    FUNCTION = "load"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "zip": (archives_in(folder_paths.get_input_directory()), {
                    "tooltip": "Zip of pictures named <anything>-<view>-<number>.png, or in front/left/back/right "
                               "folders, in ComfyUI's input directory."}),
                "index": ("INT", {"default": 0, "min": 0, "max": MAXIMUM_INDEX, "control_after_generate": True,
                                  "tooltip": "Which frame, counting from 0 in the order of their numbers."}),
                "whole_zip": ("BOOLEAN", {"default": True, "tooltip": (
                    "Queue the next frame after each run, until the zip is done. "
                    "Off, each Run does one frame.")}),
            }
        }

    @classmethod
    def IS_CHANGED(cls, zip, index, whole_zip):
        archive_path = archive_path_of(zip)
        return os.path.getmtime(archive_path) if os.path.isfile(archive_path) else zip

    def load(self, zip, index, whole_zip):
        views, number, count = numbered_views(archive_path_of(zip), index)
        print(f"[LoadNumberedViews] {index + 1}/{count} {zip}: frame {number}, {', '.join(views)}")
        picture = {"index": index, "count": count, "whole_zip": whole_zip}
        images = tuple(views.get(view) for view in VIEW_NAMES)
        return {"ui": {"picture": [picture]}, "result": (*images, number, count)}


class FrameFrontAndSideViews:
    CATEGORY = "vexel"
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "IMAGE")
    RETURN_NAMES = ("front", "left", "right", "back")
    OUTPUT_TOOLTIPS = (
        "The front view on the background, framed for Pixal3D Multi-View Conditioning.",
        "The side view when the subject faces left in it, nothing otherwise.",
        "The side view when the subject faces right in it, nothing otherwise.",
        "The back view, nothing when no back picture is connected.",
    )
    FUNCTION = "frame"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "front": ("IMAGE", {"tooltip": "The subject facing the camera."}),
                "front_mask": ("MASK", {"tooltip": "Load Image's mask of the front picture, from its alpha."}),
                "side": ("IMAGE", {"tooltip": "The subject in profile, at the same pixel scale as the front."}),
                "side_mask": ("MASK", {"tooltip": "Load Image's mask of the side picture, from its alpha."}),
                "side_faces": (["left", "right"], {
                    "tooltip": "Which way the subject faces in the side picture, choosing Pixal3D's left or right view."}),
            },
            "optional": {
                "back": ("IMAGE", {"tooltip": "The subject from behind, at the same pixel scale as the front."}),
                "back_mask": ("MASK", {"tooltip": "Load Image's mask of the back picture, from its alpha."}),
                "background": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": (
                    "Gray level behind the subject: 0 is black, as Pixal3D and Hunyuan3D shapes want, "
                    "Hunyuan3D's delight model wants about 0.8.")}),
            },
        }

    def frame(self, front, front_mask, side, side_mask, side_faces, back=None, back_mask=None, background=0.0):
        pictures = {"front": (front, front_mask), "side": (side, side_mask)}
        if back is not None:
            pictures["back"] = (back, back_mask)
        views = framed_views(pictures, background)
        print(f"[FrameFrontAndSideViews] {views['front'].shape[1]}px square, {', '.join(views)}, "
              f"side facing {side_faces}")
        left = views["side"] if side_faces == "left" else None
        right = views["side"] if side_faces == "right" else None
        return views["front"], left, right, views.get("back")


class PaintVoxelsFromViews:
    CATEGORY = "vexel"
    RETURN_TYPES = ("VOXEL",)
    FUNCTION = "paint"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "voxel": ("VOXEL", {"tooltip": "Hunyuan3D voxels, from VAEDecodeHunyuan3D."}),
                "threshold": ("FLOAT", {"default": 0.6, "min": -1.0, "max": 1.0, "step": 0.01,
                                        "tooltip": "Hunyuan3D cells above this value are solid."}),
                "front": ("IMAGE", {"tooltip": "The front picture, painting the voxels it sees."}),
                "front_mask": ("MASK",),
            },
            "optional": {
                "back": ("IMAGE", {"tooltip": "The back picture, painting the voxels it sees."}),
                "back_mask": ("MASK",),
            },
        }

    def paint(self, voxel, threshold, front, front_mask, back=None, back_mask=None):
        def picture(image, mask, name):
            return Picture(*premultiplied_image(image, mask, name))
        back_picture = picture(back, back_mask, "back") if back is not None else None
        pictures = views_of(picture(front, front_mask, "front"), back_picture)
        painted = painted_voxels(hunyuan_occupancy(voxel, threshold), pictures)
        print(f"[PaintVoxelsFromViews] {painted.data.shape[0]} surface voxels painted from {', '.join(pictures)}")
        return (painted,)


class VoxelizeTexturedMesh:
    CATEGORY = "vexel"
    RETURN_TYPES = ("VOXEL",)
    FUNCTION = "voxelize"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "trimesh": ("TRIMESH", {"tooltip": "The textured mesh from Hunyuan3DWrapper's Hy3DApplyTexture."}),
                "resolution": ("INT", {"default": 256, "min": 16, "max": 1024, "tooltip": (
                    "Voxels along the mesh's longest axis. Keep it above Save Voxel Model's maximum_size: "
                    "saving averages these into the final voxels.")}),
                "samples_per_voxel": ("INT", {"default": 8, "min": 1, "max": 64, "advanced": True, "tooltip": (
                    "Texture samples taken per voxel of surface, averaged into its color.")}),
            },
        }

    def voxelize(self, trimesh, resolution, samples_per_voxel):
        voxels = textured_mesh_voxels(trimesh, resolution, samples_per_voxel)
        print(f"[VoxelizeTexturedMesh] {voxels.data.shape[0]} surface voxels at resolution {resolution}")
        return (voxels,)
