from dataclasses import dataclass

import numpy
import torch

from .voxelextraction import face_neighbor_maximum, nearest_shell_cells
from .viewloading import bounds_of


VIEW_DIRECTIONS = {"front": (0.0, 0.0, 1.0), "back": (0.0, 0.0, -1.0)}
HORIZONTAL_DIRECTIONS = {"front": 1, "back": -1}
OPAQUE = 0.5
NORMAL_RADIUS = 2


@dataclass
class PaintedVoxels:
    data: torch.Tensor
    voxel_colors: torch.Tensor
    resolution: int


@dataclass
class Picture:
    pixels: numpy.ndarray
    alpha: numpy.ndarray


def hunyuan_occupancy(voxel, threshold):
    grid = voxel.data[0].float().cpu() > threshold
    return grid.permute(2, 1, 0).flip(0)


def surface_of(occupied):
    empty = (~occupied).to(torch.int8)
    return occupied & (face_neighbor_maximum(empty, 1) > 0)


def box_blurred(grid, radius):
    blurred = grid.float()
    kernel = 2 * radius + 1
    for axis in range(3):
        padded = torch.nn.functional.pad(blurred.movedim(axis, -1), (radius, radius))
        blurred = padded.unfold(-1, kernel, 1).mean(dim=-1).movedim(-1, axis)
    return blurred


def normals_of(occupied):
    blurred = box_blurred(occupied, NORMAL_RADIUS)
    gradients = torch.stack(torch.gradient(blurred), dim=-1)
    return -gradients / gradients.norm(dim=-1, keepdim=True).clamp(min=1e-6)


def visible_from(occupied, view):
    depth = occupied.shape[2]
    positions = torch.arange(depth).view(1, 1, depth)
    if view == "front":
        first = torch.where(occupied, positions, -1).amax(dim=2, keepdim=True)
    else:
        first = torch.where(occupied, positions, depth).amin(dim=2, keepdim=True)
    return occupied & (positions == first)


def picture_coordinates(cells, extents, view, picture):
    left, top, right, bottom = bounds_of(picture.alpha >= OPAQUE)
    low, high = extents[0]
    across = (cells[:, 0].float() - low + 0.5) / (high - low + 1)
    if HORIZONTAL_DIRECTIONS[view] < 0:
        across = 1 - across
    low_y, high_y = extents[1]
    down = 1 - (cells[:, 1].float() - low_y + 0.5) / (high_y - low_y + 1)
    columns = (left + across * (right - left)).long().clamp(0, picture.alpha.shape[1] - 1)
    rows = (top + down * (bottom - top)).long().clamp(0, picture.alpha.shape[0] - 1)
    return columns.numpy(), rows.numpy()


def colors_from(cells, extents, view, picture):
    columns, rows = picture_coordinates(cells, extents, view, picture)
    alpha = picture.alpha[rows, columns]
    opaque = alpha >= OPAQUE
    colors = picture.pixels[rows, columns] / numpy.maximum(alpha, 1e-6)[:, None]
    return torch.from_numpy(colors).float(), torch.from_numpy(opaque)


def painted_surface(occupied, pictures):
    surface = surface_of(occupied)
    cells = surface.nonzero()
    occupied_cells = occupied.nonzero()
    extents = [(int(occupied_cells[:, axis].min()), int(occupied_cells[:, axis].max())) for axis in range(3)]
    normals = normals_of(occupied)[tuple(cells.T)]
    best_score = torch.full((cells.shape[0],), -float("inf"))
    colors = torch.zeros(cells.shape[0], 3)
    for view, picture in pictures.items():
        visible = visible_from(occupied, view)[tuple(cells.T)]
        view_colors, opaque = colors_from(cells, extents, view, picture)
        score = normals @ torch.tensor(VIEW_DIRECTIONS[view])
        chosen = visible & opaque & (score > best_score)
        best_score = torch.where(chosen, score, best_score)
        colors[chosen] = view_colors[chosen]
    painted = torch.isfinite(best_score)
    return cells, colors, painted


def spread_colors(occupied, cells, colors, painted):
    owners = torch.full(occupied.shape, -1, dtype=torch.long)
    owners[tuple(cells[painted].T)] = torch.arange(int(painted.sum()))
    nearest_shell_cells(owners, occupied & (owners < 0))
    sources = owners[tuple(cells.T)]
    return torch.where((sources >= 0)[:, None], colors[painted][sources.clamp(min=0)], colors)


def views_of(front, back):
    return {"front": front} if back is None else {"front": front, "back": back}


def painted_voxels(occupied, pictures):
    if not occupied.any():
        raise ValueError("no occupied voxels, try lowering the threshold")
    occupied_cells = occupied.nonzero()
    low = occupied_cells.min(dim=0).values
    high = occupied_cells.max(dim=0).values + 1
    cropped = occupied[low[0]:high[0], low[1]:high[1], low[2]:high[2]]
    cells, colors, painted = painted_surface(cropped, pictures)
    if painted.any() and not painted.all():
        colors = spread_colors(cropped, cells, colors, painted)
    return PaintedVoxels(cells + low, colors, occupied.shape[0])
