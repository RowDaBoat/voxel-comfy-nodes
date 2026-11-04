# comfyui_vox
ComfyUI node that saves a native `VOXEL` (TRELLIS.2 and Pixal3D structures or texture voxels) as a MagicaVoxel `.vox` file.


## Install
Copy this directory into `ComfyUI/custom_nodes/` and restart ComfyUI.
The node appears as **Save Voxels as .vox** under the `vexel` category, files land in `ComfyUI/output/vox/`.
After a run the node shows a **Download** button that fetches the file to the machine running the browser.


## Wiring
- `Trellis2 VAE Decode Texture` → `voxel`: colored voxels, base color quantized into the palette.
- `Trellis2 VAE Decode Structure` → `voxel`: shape only, every voxel gets `fallback_color`.

`example_workflows/trellis2_image_to_vox.json` is the stock *Pixal3D & TRELLIS.2: Image to Model* template with both wired in.
ComfyUI lists it under the templates of this node pack.


## Inputs
- `up_axis`: texture voxels and Pixal3D structures are `y`, TRELLIS.2 structures are `z`.
- `maximum_size`: voxels along the longest axis, larger inputs are downsampled and their colors averaged, 256 at most.
- `colors`: palette size, 255 at most.
- `fallback_color`: `#rrggbb` used when the voxel carries no colors.
- `threshold`: dense grids keep cells above it.
- `batch_index`: which batch entry to save.


## Tests
`python -m unittest tests/test_vox.py`, from this directory, with `torch` installed.
