import io
import os
import re
import zipfile

import numpy
import torch
from PIL import Image, ImageOps


ALPHA_FLOOR = 32 / 255
NUMBERED_PICTURE = re.compile(r"-(\d+)\.png$", re.IGNORECASE)
ARCHIVE_EXTENSION = ".zip"


def is_picture_entry(entry):
    """Leaves out folders and the `__MACOSX/._name` resource forks macOS adds to its zips."""
    basename = os.path.basename(entry.filename)
    is_resource_fork = entry.filename.startswith("__MACOSX/") or basename.startswith("._")
    return not entry.is_dir() and not is_resource_fork and NUMBERED_PICTURE.search(basename) is not None


def number_of(entry):
    return NUMBERED_PICTURE.search(os.path.basename(entry.filename)).group(1)


def rejecting_duplicates(entries):
    entries_by_number = {}
    for entry in entries:
        number = int(number_of(entry))
        if number in entries_by_number:
            raise ValueError(f"{entries_by_number[number].filename} and {entry.filename} share the number {number}")
        entries_by_number[number] = entry
    return [entries_by_number[number] for number in sorted(entries_by_number)]


def numbered_pictures_in(archive):
    """The archive's `<anything>-<number>.png` entries in numeric order, wherever they sit in it."""
    return rejecting_duplicates([entry for entry in archive.infolist() if is_picture_entry(entry)])


def archives_in(folder):
    """Every zip under `folder`, as paths relative to it with forward slashes."""
    archives = []
    for directory, _, names in os.walk(folder):
        relative_directory = os.path.relpath(directory, folder)
        for name in names:
            if name.lower().endswith(ARCHIVE_EXTENSION):
                relative = name if relative_directory == "." else os.path.join(relative_directory, name)
                archives.append(relative.replace(os.sep, "/"))
    return sorted(archives)


def channels_of(picture):
    return numpy.asarray(picture, dtype=numpy.float32) / 255.0


def cleaned_alpha(alpha):
    """Alpha with the faint pixels cleared: generated pictures often hide colored smudges
    at alpha 1/255, invisible on screen but solid once alpha is dropped."""
    return torch.where(alpha < ALPHA_FLOOR, torch.zeros_like(alpha), alpha)


def image_and_foreground(source):
    """An IMAGE `(1,H,W,3)`, a foreground MASK `(1,H,W)` that is 1 on the subject like
    Remove Background's, and whether the picture lacked alpha and still needs its
    background removed. A picture with alpha comes out on black, its alpha as foreground."""
    with Image.open(source) as opened:
        picture = ImageOps.exif_transpose(opened)
        image = torch.from_numpy(channels_of(picture.convert("RGB")))
        has_alpha = "A" in picture.getbands() or "transparency" in picture.info
        if has_alpha:
            foreground = cleaned_alpha(torch.from_numpy(channels_of(picture.convert("RGBA").getchannel("A"))))
            image = image * foreground.unsqueeze(-1)
        else:
            foreground = torch.ones(image.shape[0:2])
    return image.unsqueeze(0), foreground.unsqueeze(0), not has_alpha


def numbered_picture(archive_path, index):
    """The `index`th numbered picture of the archive as `image_and_foreground` makes it,
    its number, how many there are, and its entry in the archive."""
    with zipfile.ZipFile(archive_path) as archive:
        entries = numbered_pictures_in(archive)
        if index >= len(entries):
            raise ValueError(f"index {index} is past the {len(entries)} numbered pictures in "
                             f"{os.path.basename(archive_path)}, the batch is done")
        image, foreground, remove_background = image_and_foreground(io.BytesIO(archive.read(entries[index])))
        return image, foreground, remove_background, number_of(entries[index]), len(entries), entries[index].filename
