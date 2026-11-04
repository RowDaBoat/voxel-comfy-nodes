from .nodes import LoadNumberedImage, LoadNumberedViews, SaveVoxelsAsVox, SaveVoxelsToVoxCollection


NODE_CLASS_MAPPINGS = {
    "SaveVoxelsAsVox": SaveVoxelsAsVox,
    "SaveVoxelsToVoxCollection": SaveVoxelsToVoxCollection,
    "LoadNumberedImage": LoadNumberedImage,
    "LoadNumberedViews": LoadNumberedViews,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveVoxelsAsVox": "Save Voxels as .vox",
    "SaveVoxelsToVoxCollection": "Save Voxels to Multi-Model .vox",
    "LoadNumberedImage": "Load Numbered Image",
    "LoadNumberedViews": "Load Numbered Views",
}
WEB_DIRECTORY = "./web"
