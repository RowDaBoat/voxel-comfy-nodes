import os
import re


DIGITS = re.compile(r"(\d+)")
UNSAFE_CHARACTERS = re.compile(r"[^\w\-]+")


def natural_key(name):
    """Sorts `2.png` before `10.png`, the order numbered pictures are meant in."""
    return [int(part) if part.isdigit() else part.lower() for part in DIGITS.split(name)]


def naturally_sorted(names):
    return sorted(names, key=natural_key)


def stem_of(path):
    return os.path.splitext(os.path.basename(path))[0]


def safe_name(name, fallback):
    cleaned = UNSAFE_CHARACTERS.sub("_", name).strip("_")
    return cleaned or fallback


def folder_inside(root, folder):
    """`folder` resolved under `root`, refusing paths that climb out of it."""
    resolved_root = os.path.realpath(root)
    resolved = os.path.realpath(os.path.join(resolved_root, folder))
    if os.path.commonpath((resolved_root, resolved)) != resolved_root:
        raise ValueError(f"{folder!r} is outside of {resolved_root}")
    return resolved
