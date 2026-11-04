# comfyui_vox
ComfyUI nodes that save a native `VOXEL` (TRELLIS.2 and Pixal3D structures or texture voxels) as MagicaVoxel `.vox` files, one model per file or many models in one.


## Install
Copy this directory into `ComfyUI/custom_nodes/` and restart ComfyUI.
The nodes appear under the `vexel` category:
- **Save Voxels as .vox**: one model per file, saved to `ComfyUI/output/vox/`.
- **Save Voxels to Multi-Model .vox**: adds a model to a collection and rewrites the collection's multi-model file.
- **Load Numbered Image**: one picture of an uploaded zip per run, in the order of their numbers.

Both save nodes have a **Download** button that fetches their latest file to the machine running the browser.


## Wiring
- `Trellis2 VAE Decode Texture` → `voxel`: colored voxels, base color quantized into the palette.
- `Trellis2 VAE Decode Structure` → `voxel`: shape only, every voxel gets `fallback_color`.

`example_workflows/trellis2_image_to_vox.json` is the stock *Pixal3D & TRELLIS.2: Image to Model* template with both wired in.
ComfyUI lists it under the templates of this node pack.


## Bulk
`example_workflows/trellis2_images_to_multi_model_vox.json` turns a zip of numbered pictures into one `.vox`:
1. Zip the pictures, named `<anything>-<number>.png` (`hero-1.png`, `hero-2.png`, ... `hero-10.png`), anywhere in the zip.
2. Press **Upload zip** on Load Numbered Image, which also resets `index` to `0`.
3. Press **Run** once. With `whole_zip` on, each picture queues the next after it until the zip is done, as long as the window stays open; off, each Run does one picture.

Other files in the zip, and the `__MACOSX` entries macOS adds, are ignored; two pictures with the same number are refused.
A picture with alpha is its own cutout: it comes out on black with its alpha as `foreground`, and `remove_background` is false so the workflow skips Remove Background.
Pixels below alpha 32/255 are dropped first, generated pictures hide colored smudges there that turn solid once alpha is ignored.
Each run processes one picture end to end, so memory stays flat however many there are, and a run past the last picture stops with an error.
The next picture is queued as soon as one loads, so a picture failing later on does not stop the ones after it.
The model is saved as `output/<collection>/<number>.vox`, then `output/<collection>.vox` is rewritten with every model of the collection: named by its picture's number, on a grid standing on z = 0, all sharing one palette of `colors`.
A picture run again replaces its model; a new set of pictures wants a new `collection`.

A zip can also be uploaded through ComfyUI's API, and is then listed by the loader:
```sh
curl -F "image=@heroes.zip" -F overwrite=true http://<host>:8188/upload/image
```


## Inputs
- `up_axis`: texture voxels and Pixal3D structures are `y`, TRELLIS.2 structures are `z`.
- `maximum_size`: voxels along the longest axis, larger inputs are downsampled and their colors averaged, 256 at most.
- `colors`: palette size, 255 at most.
- `fallback_color`: `#rrggbb` used when the voxel carries no colors.
- `threshold`: dense grids keep cells above it.
- `batch_index`: which batch entry to save.
- `name` (multi-model): the model's name, wired from the loader; empty numbers it after the collection's models.
- `collection` (multi-model): where the models and the multi-model file go, inside the output directory.
- `spacing` (multi-model): empty voxels between models.


## Tests
`python -m unittest tests/test_vox.py`, from this directory, with `torch` installed.
