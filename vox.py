import struct
from collections import namedtuple


VOX_VERSION = 150
MAXIMUM_DIMENSION = 256
PALETTE_LENGTH = 256
ROOT_TRANSFORM_ID = 0
ROOT_GROUP_ID = 1
FIRST_MODEL_NODE_ID = 2
NO_LAYER = -1
MODEL_LAYER = 0
RESERVED = -1


VoxModel = namedtuple("VoxModel", ("name", "size", "voxels", "translation"))


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


def text(value):
    encoded = value.encode("utf-8")
    return struct.pack("<i", len(encoded)) + encoded


def dictionary(pairs):
    content = struct.pack("<i", len(pairs))
    for key, value in pairs.items():
        content += text(key) + text(value)
    return content


def transform_chunk(node_id, child_id, layer_id, attributes, frame):
    content = struct.pack("<i", node_id) + dictionary(attributes)
    content += struct.pack("<iiii", child_id, RESERVED, layer_id, 1) + dictionary(frame)
    return chunk(b"nTRN", content)


def group_chunk(node_id, child_ids):
    content = struct.pack("<i", node_id) + dictionary({}) + struct.pack("<i", len(child_ids))
    content += b"".join(struct.pack("<i", child_id) for child_id in child_ids)
    return chunk(b"nGRP", content)


def shape_chunk(node_id, model_index):
    content = struct.pack("<i", node_id) + dictionary({}) + struct.pack("<ii", 1, model_index) + dictionary({})
    return chunk(b"nSHP", content)


def layer_chunk(layer_id):
    return chunk(b"LAYR", struct.pack("<i", layer_id) + dictionary({}) + struct.pack("<i", RESERVED))


def model_chunks(size, voxels):
    validate_size(size)
    for voxel in voxels:
        validate_voxel(voxel, size)
    return size_chunk(size) + voxels_chunk(voxels)


def translation_text(translation):
    return " ".join(str(int(component)) for component in translation)


def scene_chunks(models):
    """Root transform, a group, and one named and translated transform with its shape per model."""
    transform_ids = [FIRST_MODEL_NODE_ID + 2 * index for index in range(len(models))]
    chunks = transform_chunk(ROOT_TRANSFORM_ID, ROOT_GROUP_ID, NO_LAYER, {}, {})
    chunks += group_chunk(ROOT_GROUP_ID, transform_ids)
    for index, (model, transform_id) in enumerate(zip(models, transform_ids)):
        frame = {"_t": translation_text(model.translation)}
        chunks += transform_chunk(transform_id, transform_id + 1, MODEL_LAYER, {"_name": model.name}, frame)
        chunks += shape_chunk(transform_id + 1, index)
    return chunks + layer_chunk(MODEL_LAYER)


def file_bytes(children):
    return b"VOX " + struct.pack("<i", VOX_VERSION) + chunk(b"MAIN", b"", children)


def vox_bytes(size, voxels, palette):
    return file_bytes(model_chunks(size, voxels) + palette_chunk(padded_palette(palette)))


def multi_model_vox_bytes(models, palette):
    if not models:
        raise ValueError("a multi-model vox needs at least one model")
    children = b"".join(model_chunks(model.size, model.voxels) for model in models)
    return file_bytes(children + scene_chunks(models) + palette_chunk(padded_palette(palette)))


def write_vox(path, size, voxels, palette):
    with open(path, "wb") as file:
        file.write(vox_bytes(size, voxels, palette))


def write_multi_model_vox(path, models, palette):
    with open(path, "wb") as file:
        file.write(multi_model_vox_bytes(models, palette))


def chunks_of(content):
    offset = 0
    while offset < len(content):
        identifier = content[offset:offset + 4]
        content_length, children_length = struct.unpack_from("<ii", content, offset + 4)
        yield identifier, content[offset + 12:offset + 12 + content_length]
        offset += 12 + content_length + children_length


def first_model_of(content):
    """Size, voxels and palette of a file's first model, as written by `vox_bytes`."""
    if content[:4] != b"VOX ":
        raise ValueError("missing the VOX header")
    _, children_length = struct.unpack_from("<ii", content, 12)
    chunks = {}
    for identifier, chunk_content in chunks_of(content[20:20 + children_length]):
        chunks.setdefault(identifier, chunk_content)
    size = struct.unpack("<iii", chunks[b"SIZE"])
    voxel_bytes = chunks[b"XYZI"][4:]
    voxels = [tuple(voxel_bytes[offset:offset + 4]) for offset in range(0, len(voxel_bytes), 4)]
    palette_bytes = chunks[b"RGBA"]
    palette = [tuple(palette_bytes[offset:offset + 4]) for offset in range(0, PALETTE_LENGTH * 4, 4)]
    return size, voxels, palette


def read_vox(path):
    with open(path, "rb") as file:
        return first_model_of(file.read())


def color_from_hex(hex_color):
    digits = hex_color.strip().lstrip("#")
    if len(digits) != 6:
        raise ValueError(f"color must look like #rrggbb, got {hex_color!r}")
    return (int(digits[0:2], 16), int(digits[2:4], 16), int(digits[4:6], 16), 255)


def single_color_palette(hex_color):
    return [color_from_hex(hex_color)]
