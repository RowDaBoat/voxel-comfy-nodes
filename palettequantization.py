import torch


REFINEMENT_ITERATIONS = 16
SEED = 0


def squared_distances(colors, centers):
    return torch.cdist(colors, centers) ** 2


def initial_centers(colors, count, generator):
    first = torch.randint(colors.shape[0], (1,), generator=generator)
    centers = [colors[first]]
    distances = squared_distances(colors, centers[0]).squeeze(1)
    for _ in range(count - 1):
        chosen = torch.multinomial(distances, 1, generator=generator)
        centers.append(colors[chosen])
        distances = torch.minimum(distances, squared_distances(colors, colors[chosen]).squeeze(1))
    return torch.cat(centers)


def refined_centers(colors, centers):
    assignments = squared_distances(colors, centers).argmin(dim=1)
    sums = torch.zeros_like(centers).index_add_(0, assignments, colors)
    counts = torch.bincount(assignments, minlength=centers.shape[0]).unsqueeze(1)
    return torch.where(counts > 0, sums / counts.clamp(min=1), centers)


def clustered(colors, count):
    generator = torch.Generator().manual_seed(SEED)
    centers = initial_centers(colors, count, generator)
    for _ in range(REFINEMENT_ITERATIONS):
        centers = refined_centers(colors, centers)
    return centers.round(), squared_distances(colors, centers.round()).argmin(dim=1)


def quantized(colors, count):
    eight_bit_colors = (colors.clamp(0, 1) * 255).round()
    unique_colors, indices = torch.unique(eight_bit_colors, dim=0, return_inverse=True)
    if unique_colors.shape[0] <= count:
        return unique_colors, indices
    return clustered(eight_bit_colors, count)


def palette_of(centers):
    return [(red, green, blue, 255) for red, green, blue in centers.int().tolist()]
