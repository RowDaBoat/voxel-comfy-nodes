from .nodes import FrameFrontAndSideViews, LoadNumberedImage, LoadNumberedViews, PaintVoxelsFromViews, SaveVoxelsAsVox, SaveVoxelsToVoxCollection, VoxelizeTexturedMesh


NODE_CLASS_MAPPINGS = {
    "SaveVoxelsAsVox": SaveVoxelsAsVox,
    "SaveVoxelsToVoxCollection": SaveVoxelsToVoxCollection,
    "LoadNumberedImage": LoadNumberedImage,
    "LoadNumberedViews": LoadNumberedViews,
    "FrameFrontAndSideViews": FrameFrontAndSideViews,
    "PaintVoxelsFromViews": PaintVoxelsFromViews,
    "VoxelizeTexturedMesh": VoxelizeTexturedMesh,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "SaveVoxelsAsVox": "Save Voxels as .vox",
    "SaveVoxelsToVoxCollection": "Save Voxels to Multi-Model .vox",
    "LoadNumberedImage": "Load Numbered Image",
    "LoadNumberedViews": "Load Numbered Views",
    "FrameFrontAndSideViews": "Frame Front, Side and Back Views",
    "PaintVoxelsFromViews": "Paint Hunyuan3D Voxels from Views",
    "VoxelizeTexturedMesh": "Voxelize Textured Hunyuan3D Mesh",
}
WEB_DIRECTORY = "./web"
