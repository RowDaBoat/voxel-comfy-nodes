import torch


BASE_COLOR_CHANNELS = slice(0, 3)


def dense_points(grid, batch_index, threshold):
    occupied = grid[batch_index] > threshold
    return occupied.nonzero().long(), None


def sparse_points(coordinates, colors, batch_index):
    rows = coordinates.long()
    if rows.shape[1] == 3:
        return rows, colors
    selected = rows[:, 0] == batch_index
    selected_colors = colors[selected] if colors is not None else None
    return rows[selected][:, 1:], selected_colors


def base_colors_of(voxel):
    if voxel.voxel_colors is None:
        return None
    return voxel.voxel_colors[:, BASE_COLOR_CHANNELS].float().cpu()


def voxel_points(voxel, batch_index, threshold):
    data = voxel.data.cpu()
    if data.dim() == 4:
        return dense_points(data, batch_index, threshold)
    if data.dim() == 2 and data.shape[1] in (3, 4):
        return sparse_points(data, base_colors_of(voxel), batch_index)
    raise ValueError(f"unsupported voxel data shape {tuple(data.shape)}")


def rotated_to_z_up(coordinates):
    x, y, z = coordinates.unbind(dim=1)
    return torch.stack((x, -z, y), dim=1)


def rotated_to_z_up_in_grid(coordinates, resolution):
    """The same rotation, kept inside the `resolution` grid: undoes the decoder's
    `(x, y, z) -> (x, z, R - 1 - y)`."""
    x, y, z = coordinates.unbind(dim=1)
    return torch.stack((x, resolution - 1 - z, y), dim=1)


def grid_resolution_of(voxel):
    """The side of the grid TRELLIS generated in: dense grids are that size, sparse
    voxels carry it, or it is the smallest power of two holding them."""
    data = voxel.data
    if data.dim() == 4:
        return data.shape[-1]
    if voxel.resolution is not None:
        return int(voxel.resolution)
    extent = int(data[:, -3:].max().item()) + 1
    return 1 << max(0, (extent - 1).bit_length())


def averaged(colors, cell_indices, cell_count):
    if colors is None:
        return None
    sums = torch.zeros(cell_count, colors.shape[1]).index_add_(0, cell_indices, colors)
    counts = torch.bincount(cell_indices, minlength=cell_count).unsqueeze(1)
    return sums / counts


def fitted(coordinates, colors, maximum_size):
    if coordinates.shape[0] == 0:
        raise ValueError("no occupied voxels, try lowering the threshold")
    offsets = coordinates - coordinates.min(dim=0).values
    extent = int(offsets.max().item()) + 1
    scale = min(1.0, maximum_size / extent)
    cells = torch.floor(offsets.float() * scale).long()
    unique_cells, cell_indices = torch.unique(cells, dim=0, return_inverse=True)
    return unique_cells, averaged(colors, cell_indices, unique_cells.shape[0])


def size_of(cells):
    return tuple((cells.max(dim=0).values + 1).tolist())


def placed_in_grid(coordinates, colors, maximum_size, resolution):
    """Cells of a `maximum_size` grid scaled from the whole TRELLIS grid, so models of
    different pictures keep their place relative to each other."""
    scale = min(1.0, maximum_size / resolution)
    cells = torch.floor(coordinates.float() * scale).long().clamp(0, maximum_size - 1)
    if cells.shape[0] == 0:
        raise ValueError("no occupied voxels, try lowering the threshold")
    unique_cells, cell_indices = torch.unique(cells, dim=0, return_inverse=True)
    return unique_cells, averaged(colors, cell_indices, unique_cells.shape[0])


def grid_size_of(maximum_size, resolution):
    side = min(maximum_size, resolution)
    return (side, side, side)
