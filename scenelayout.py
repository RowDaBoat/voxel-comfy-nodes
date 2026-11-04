import math


def grid_translations(sizes, spacing):
    """Model centers on a square grid of equal cells, every model standing on z = 0.

    MagicaVoxel places a model's voxel `v` at `translation + v - size // 2`.
    """
    columns = max(1, math.ceil(math.sqrt(len(sizes))))
    cell_width = max(size[0] for size in sizes) + spacing
    cell_depth = max(size[1] for size in sizes) + spacing
    translations = []
    for index, (width, depth, height) in enumerate(sizes):
        row, column = divmod(index, columns)
        x = column * cell_width + width // 2
        y = -row * cell_depth + depth // 2
        translations.append((x, y, height // 2))
    return translations
