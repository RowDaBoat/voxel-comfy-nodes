import io
import os
import re
import zipfile

import numpy
import torch
from PIL import Image, ImageOps


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


def image_and_mask(source):
    """An IMAGE `(1,H,W,3)` and a MASK `(1,H,W)` the way ComfyUI's Load Image makes them:
    the mask is one minus alpha, empty when the picture has no alpha."""
    with Image.open(source) as opened:
        picture = ImageOps.exif_transpose(opened)
        image = torch.from_numpy(channels_of(picture.convert("RGB"))).unsqueeze(0)
        if "A" in picture.getbands():
            alpha = torch.from_numpy(channels_of(picture.getchannel("A")))
            mask = 1.0 - alpha
        else:
            mask = torch.zeros(image.shape[1:3])
    return image, mask.unsqueeze(0)


def numbered_picture(archive_path, index):
    """The `index`th numbered picture of the archive, its number, and how many there are."""
    with zipfile.ZipFile(archive_path) as archive:
        entries = numbered_pictures_in(archive)
        if index >= len(entries):
            raise ValueError(f"index {index} is past the {len(entries)} numbered pictures in "
                             f"{os.path.basename(archive_path)}, the batch is done")
        image, mask = image_and_mask(io.BytesIO(archive.read(entries[index])))
        return image, mask, number_of(entries[index]), len(entries), entries[index].filename
