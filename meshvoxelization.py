import math

import numpy
import torch

from .voxelextraction import averaged
from .voxelpainting import PaintedVoxels


SEED = 0


def texture_of(mesh):
    material = getattr(mesh.visual, "material", None)
    image = getattr(material, "image", None) or getattr(material, "baseColorTexture", None)
    if image is None or getattr(mesh.visual, "uv", None) is None:
        raise ValueError("the mesh has no texture, connect Hy3DApplyTexture's textured mesh")
    return numpy.asarray(image.convert("RGB"), dtype=numpy.float32) / 255.0


def hunyuan_to_canonical(vertices):
    front, up, right = vertices[:, 0], vertices[:, 1], vertices[:, 2]
    return numpy.stack((-right, up, front), axis=1)


def texture_colors(texture, uv):
    height, width = texture.shape[:2]
    clamped = numpy.clip(uv, 0.0, 1.0)
    columns = numpy.rint(clamped[:, 0] * (width - 1)).astype(numpy.int64)
    rows = numpy.rint((1.0 - clamped[:, 1]) * (height - 1)).astype(numpy.int64)
    return texture[rows, columns]


def face_areas(corners):
    return 0.5 * numpy.linalg.norm(numpy.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1)


def surface_samples(corners, areas, count, generator):
    faces = numpy.arange(corners.shape[0])
    weights = numpy.full((corners.shape[0], 3), 1.0 / 3.0)
    if count > 0 and areas.sum() > 0:
        chosen = generator.choice(corners.shape[0], count, p=areas / areas.sum())
        spread = numpy.sqrt(generator.random(count))
        along = generator.random(count)
        faces = numpy.concatenate((faces, chosen))
        weights = numpy.concatenate((weights, numpy.stack((1 - spread, spread * (1 - along), spread * along), axis=1)))
    return faces, weights


def textured_mesh_voxels(mesh, resolution, samples_per_voxel):
    texture = texture_of(mesh)
    vertices = hunyuan_to_canonical(numpy.asarray(mesh.vertices, dtype=numpy.float64))
    faces = numpy.asarray(mesh.faces, dtype=numpy.int64)
    uv = numpy.asarray(mesh.visual.uv, dtype=numpy.float64)
    low = vertices.min(axis=0)
    size = (vertices.max(axis=0) - low).max() / resolution
    corners = vertices[faces]
    areas = face_areas(corners)
    count = math.ceil(areas.sum() / size ** 2 * samples_per_voxel)
    sampled_faces, weights = surface_samples(corners, areas, count, numpy.random.default_rng(SEED))
    points = numpy.einsum("nk,nkd->nd", weights, corners[sampled_faces])
    colors = texture_colors(texture, numpy.einsum("nk,nkd->nd", weights, uv[faces][sampled_faces]))
    cells = numpy.clip(numpy.floor((points - low) / size), 0, resolution - 1).astype(numpy.int64)
    unique_cells, cell_indices = torch.unique(torch.from_numpy(cells), dim=0, return_inverse=True)
    cell_colors = averaged(torch.from_numpy(colors), cell_indices, unique_cells.shape[0])
    return PaintedVoxels(unique_cells, cell_colors, resolution)
