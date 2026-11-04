import os

import torch

from .naming import naturally_sorted, stem_of
from .palettequantization import quantized, palette_of
from .scenelayout import grid_translations
from .vox import VoxModel, read_vox, write_multi_model_vox


MODEL_EXTENSION = ".vox"


def model_paths_in(folder):
    names = [name for name in os.listdir(folder) if name.lower().endswith(MODEL_EXTENSION)]
    return [os.path.join(folder, name) for name in naturally_sorted(names)]


def colors_of(voxels, palette):
    rows = [palette[color_index - 1][:3] for *_, color_index in voxels]
    return torch.tensor(rows, dtype=torch.float32) / 255.0


def recolored(voxels, indices):
    return [(x, y, z, index + 1) for (x, y, z, _), index in zip(voxels, indices)]


def collected_models(paths, color_count, spacing):
    """Every model file merged into one scene, their colors quantized into one shared palette."""
    read_models = [(stem_of(path), *read_vox(path)) for path in paths]
    colors = torch.cat([colors_of(voxels, palette) for _, _, voxels, palette in read_models])
    centers, indices = quantized(colors, color_count)
    translations = grid_translations([size for _, size, _, _ in read_models], spacing)
    models = []
    offset = 0
    for (name, size, voxels, _), translation in zip(read_models, translations):
        model_indices = indices[offset:offset + len(voxels)].tolist()
        models.append(VoxModel(name, size, recolored(voxels, model_indices), translation))
        offset += len(voxels)
    return models, palette_of(centers)


def write_collection(models_folder, path, color_count, spacing):
    models, palette = collected_models(model_paths_in(models_folder), color_count, spacing)
    write_multi_model_vox(path, models, palette)
    return models, palette
