import struct


VOX_VERSION = 150
MAXIMUM_DIMENSION = 256
PALETTE_LENGTH = 256


def chunk(identifier, content, children=b""):
    return identifier + struct.pack("<ii", len(content), len(children)) + content + children


def size_chunk(size):
    return chunk(b"SIZE", struct.pack("<iii", *size))


def voxels_chunk(voxels):
    content = bytearray(struct.pack("<i", len(voxels)))
    for x, y, z, color_index in voxels:
        content += bytes((x, y, z, color_index))
    return chunk(b"XYZI", bytes(content))


def palette_chunk(palette):
    content = bytearray()
    for red, green, blue, alpha in palette:
        content += bytes((red, green, blue, alpha))
    return chunk(b"RGBA", bytes(content))


def validate_size(size):
    for dimension in size:
        if not 1 <= dimension <= MAXIMUM_DIMENSION:
            raise ValueError(f"vox models are limited to {MAXIMUM_DIMENSION} per axis, got {tuple(size)}")


def validate_voxel(voxel, size):
    x, y, z, color_index = voxel
    inside = 0 <= x < size[0] and 0 <= y < size[1] and 0 <= z < size[2]
    if not inside:
        raise ValueError(f"voxel {(x, y, z)} is outside of size {tuple(size)}")
    if not 1 <= color_index <= 255:
        raise ValueError(f"color index {color_index} must be in 1..255")


def padded_palette(palette):
    if len(palette) > PALETTE_LENGTH:
        raise ValueError(f"palette has {len(palette)} colors, maximum is {PALETTE_LENGTH}")
    return list(palette) + [(0, 0, 0, 255)] * (PALETTE_LENGTH - len(palette))


def vox_bytes(size, voxels, palette):
    validate_size(size)
    for voxel in voxels:
        validate_voxel(voxel, size)
    children = size_chunk(size) + voxels_chunk(voxels) + palette_chunk(padded_palette(palette))
    return b"VOX " + struct.pack("<i", VOX_VERSION) + chunk(b"MAIN", b"", children)


def write_vox(path, size, voxels, palette):
    with open(path, "wb") as file:
        file.write(vox_bytes(size, voxels, palette))


def color_from_hex(hex_color):
    digits = hex_color.strip().lstrip("#")
    if len(digits) != 6:
        raise ValueError(f"color must look like #rrggbb, got {hex_color!r}")
    return (int(digits[0:2], 16), int(digits[2:4], 16), int(digits[4:6], 16), 255)


def single_color_palette(hex_color):
    return [color_from_hex(hex_color)]
