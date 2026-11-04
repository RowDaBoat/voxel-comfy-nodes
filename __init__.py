from .nodes import LoadNumberedImage, SaveVoxelsAsVox, SaveVoxelsToVoxCollection


NODE_CLASS_MAPPINGS = {
    "SaveVoxelsAsVox": SaveVoxelsAsVox,
    "SaveVoxelsToVoxCollection": SaveVoxelsToVoxCollection,
    "LoadNumberedImage": LoadNumberedImage,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveVoxelsAsVox": "Save Voxels as .vox",
    "SaveVoxelsToVoxCollection": "Save Voxels to Multi-Model .vox",
    "LoadNumberedImage": "Load Numbered Image",
}
WEB_DIRECTORY = "./web"
