import io
import math
import os
import re
import zipfile

import numpy
import torch
from PIL import Image, ImageOps

from .imageloading import channels_of, cleaned_alpha, is_picture_entry, number_of


VIEW_NAMES = ("front", "left", "back", "right")
WORD_SEPARATORS = re.compile(r"[^a-z]+")
FRAMING_PAD = 1.1


def view_of(entry):
    words = set(WORD_SEPARATORS.split(entry.filename.lower()))
    views = [view for view in VIEW_NAMES if view in words]
    if len(views) != 1:
        raise ValueError(f"{entry.filename} should name exactly one view out of {', '.join(VIEW_NAMES)}, "
                         f"in its name or its folder")
    return views[0]


def add_view(frames, entry):
    number = int(number_of(entry))
    view = view_of(entry)
    frame = frames.setdefault(number, {})
    if view in frame:
        raise ValueError(f"{frame[view].filename} and {entry.filename} are both the {view} view of frame {number}")
    frame[view] = entry


def views_of(frames):
    return [view for view in VIEW_NAMES if any(view in frame for frame in frames.values())]


def require_every_view(frames, views):
    for number, frame in frames.items():
        missing = [view for view in views if view not in frame]
        if missing:
            raise ValueError(f"frame {number} has no {', '.join(missing)} view")


def numbered_frames_in(archive):
    """Each frame's views, `{view: entry}`, in the order of the frame numbers."""
    frames = {}
    for entry in archive.infolist():
        if is_picture_entry(entry):
            add_view(frames, entry)
    require_every_view(frames, views_of(frames))
    return [frames[number] for number in sorted(frames)]


def premultiplied_picture(archive, entry):
    """Colors on black and the cleaned alpha, `(H,W,3)` and `(H,W)`."""
    with Image.open(io.BytesIO(archive.read(entry))) as opened:
        picture = ImageOps.exif_transpose(opened).convert("RGBA")
        pixels = channels_of(picture)
    alpha = cleaned_alpha(torch.from_numpy(pixels[..., 3])).numpy()
    if alpha.max() == 1.0 and alpha.min() == 1.0:
        raise ValueError(f"{entry.filename} has no transparency, multi-view pictures need their background in alpha")
    return pixels[..., :3] * alpha[..., None], alpha


def bounds_of(alpha):
    rows = numpy.nonzero(alpha.any(axis=1))[0]
    columns = numpy.nonzero(alpha.any(axis=0))[0]
    if rows.size == 0:
        return None
    return columns[0], rows[0], columns[-1] + 1, rows[-1] + 1


def union_of(bounds):
    present = [bound for bound in bounds if bound is not None]
    if not present:
        return None
    return (min(bound[0] for bound in present), min(bound[1] for bound in present),
            max(bound[2] for bound in present), max(bound[3] for bound in present))


def view_bounds_of(archive, frames, view):
    return union_of([bounds_of(premultiplied_picture(archive, frame[view])[1]) for frame in frames])


def square_framing(bounds):
    """One square per view, all of the same side: each view centered horizontally on its
    bounds, every view at the same height, the widest extent filling 1/1.1 of the side,
    the rig Pixal3D expects."""
    top = min(bound[1] for bound in bounds.values())
    bottom = max(bound[3] for bound in bounds.values())
    widest = max(max(bound[2] - bound[0] for bound in bounds.values()), bottom - top)
    side = math.ceil(widest * FRAMING_PAD)
    middle = (top + bottom) / 2
    return {view: (round((bound[0] + bound[2]) / 2 - side / 2), round(middle - side / 2), side)
            for view, bound in bounds.items()}


def framing_of(archive, frames):
    views = views_of({index: frame for index, frame in enumerate(frames)})
    bounds = {view: view_bounds_of(archive, frames, view) for view in views}
    if any(bound is None for bound in bounds.values()):
        raise ValueError("a view is fully transparent in every frame")
    return square_framing(bounds)


def cropped(pixels, left, top, side):
    """A `side` square of `pixels` from `(left, top)`, black where it reaches past them."""
    square = numpy.zeros((side, side) + pixels.shape[2:], dtype=numpy.float32)
    height, width = pixels.shape[:2]
    source_left, source_top = max(left, 0), max(top, 0)
    source_right, source_bottom = min(left + side, width), min(top + side, height)
    if source_right > source_left and source_bottom > source_top:
        square[source_top - top:source_bottom - top, source_left - left:source_right - left] = \
            pixels[source_top:source_bottom, source_left:source_right]
    return square


framings = {}


def cached_framing(archive_path, archive, frames):
    key = (archive_path, os.path.getmtime(archive_path))
    if key not in framings:
        framings.clear()
        framings[key] = framing_of(archive, frames)
    return framings[key]


def framed_view(archive, entry, square):
    pixels, _ = premultiplied_picture(archive, entry)
    return torch.from_numpy(cropped(pixels, *square)).unsqueeze(0)


def numbered_views(archive_path, index):
    """The `index`th frame's views as IMAGEs `(1,S,S,3)` keyed by view, its number, and how
    many frames there are."""
    with zipfile.ZipFile(archive_path) as archive:
        frames = numbered_frames_in(archive)
        if index >= len(frames):
            raise ValueError(f"index {index} is past the {len(frames)} frames in "
                             f"{os.path.basename(archive_path)}, the batch is done")
        framing = cached_framing(archive_path, archive, frames)
        frame = frames[index]
        views = {view: framed_view(archive, entry, framing[view]) for view, entry in frame.items()}
        number = number_of(next(iter(frame.values())))
        return views, number, len(frames)


def premultiplied_image(image, mask, name):
    pixels = image[0, ..., :3].float().cpu().numpy()
    if mask is None or tuple(mask.shape[-2:]) != tuple(pixels.shape[:2]):
        raise ValueError(f"the {name} picture has no transparency, connect its mask and give it its background in alpha")
    alpha = cleaned_alpha(1.0 - mask.reshape(-1, *pixels.shape[:2])[0].float().cpu()).numpy()
    if alpha.min() == 1.0:
        raise ValueError(f"the {name} picture has no transparency, it needs its background in alpha")
    if alpha.max() == 0.0:
        raise ValueError(f"the {name} picture is fully transparent")
    return pixels * alpha[..., None], alpha


def on_background(pixels, alpha, square, background):
    framed = cropped(pixels, *square)
    if background:
        framed = framed + background * (1.0 - cropped(alpha, *square)[..., None])
    return torch.from_numpy(framed).unsqueeze(0)


def framed_views(pictures, background=0.0):
    premultiplied = {name: premultiplied_image(image, mask, name) for name, (image, mask) in pictures.items()}
    framing = square_framing({name: bounds_of(alpha) for name, (_, alpha) in premultiplied.items()})
    return {name: on_background(pixels, alpha, framing[name], background)
            for name, (pixels, alpha) in premultiplied.items()}
