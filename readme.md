# comfyui_vox
ComfyUI nodes that save a native `VOXEL` (TRELLIS.2 and Pixal3D structures or texture voxels) as MagicaVoxel `.vox` files, one model per file or many models in one.


## Install
Copy this directory into `ComfyUI/custom_nodes/` and restart ComfyUI.
The nodes appear under the `vexel` category:
- **Save Voxels as .vox**: one model per file, saved to `ComfyUI/output/vox/`.
- **Save Voxels to Multi-Model .vox**: adds a model to a collection and rewrites the collection's multi-model file.
- **Load Numbered Image**: one picture of an uploaded zip per run, in the order of their numbers.
- **Load Numbered Views**: every view of one frame of an uploaded zip per run, framed for Pixal3D Multi-View Conditioning.
- **Frame Front, Side and Back Views**: a front, a side and an optional back picture from Load Image nodes, framed for Pixal3D Multi-View Conditioning.
- **Paint Hunyuan3D Voxels from Views**: colors a Hunyuan3D shape from the front, side and back pictures it was made from.
- **Voxelize Textured Hunyuan3D Mesh**: turns a mesh painted by Hunyuan3DWrapper into colored voxels, averaging several texture samples into each.

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


## Multi-view
`example_workflows/pixal3d_multiview_frames_to_vox.json` generates each frame from several views of it, so the frames hold together from every angle, not only the one they were drawn from:
1. Zip every view of every frame as `<anything>-<view>-<number>.png`, or inside `front/`, `left/`, `back/` and `right/` folders. Use 2 to 4 views, the same ones in every frame, with the background in alpha and the same pixel scale in every picture.
2. Press **Upload zip** on Load Numbered Views and press **Run** once.

Load Numbered Views frames the views the way Pixal3D's rig expects, 90 degrees apart around the subject: one square side for every view and frame, the widest extent of the whole animation filling 1/1.1 of it, every view at the same height, each centered on where the subject moves in it.
Sharing the framing across frames keeps their scale and position, which per-frame cropping would lose.
Both save nodes use `placement: grid` for the same reason.



## Front, side and back
`example_workflows/pixal3d_front_side_to_vox.json` makes one model from a front, a side and a back picture:
1. Upload the front on **Front View**, the profile on **Side View** and the subject from behind on **Back View**, all with the background in alpha and at the same pixel scale. Back View starts bypassed, since Pixal3D tends to mix a back picture's details with the front's; press Ctrl+B on it to try one.
2. Set **side_faces** to the way the subject faces in the side picture: `left` feeds Pixal3D's left view, `right` its right view.
3. Set **maximum_size** on **Save Voxel Model** to the voxels along the longest axis, then **Run** and **Download**.

Every Run uses new seeds; set the KSamplers' control_after_generate to fixed to repeat one.

maximum_size only scales down, so a model past TRELLIS's own resolution stays at that resolution; raise Trellis2UpsampleStage's resolution for more voxels.

## Hunyuan3D-2mv
`example_workflows/hunyuan3d_mv_to_vox.json` makes the shape from the same front, side and back pictures with Hunyuan3D-2mv, which takes those views natively and keeps the back from mixing into the front:
1. Upload the pictures as for the Pixal3D template and keep **side_faces** on `left`. Without a back picture, bypass **Back View** and **Encode Back** with Ctrl+B.
2. Set **maximum_size** on **Save Voxel Model**, then **Run** and **Download**. **Preview Mesh** shows the shape.

Hunyuan3D-2mv makes the shape only; **Paint Hunyuan3D Voxels from Views** colors it from the pictures. Each surface voxel takes the pixel of the picture it faces most, among the pictures that see it unblocked; the side picture paints both sides, mirrored on the far one; voxels no picture sees, such as the top of the head or the soles, take the nearest painted color. It also turns the model to face MagicaVoxel's front. Keep its **side_faces** the same as the framing node's.

It needs `hunyuan3d-dit-v2-mv-turbo_fp16.safetensors` in `models/checkpoints`; the KSampler is set for it, 10 steps at cfg 1. The full `hunyuan3d-dit-v2-mv_fp16.safetensors` wants about 30 steps at cfg 5.

## Hunyuan3D-Paint
`example_workflows/hunyuan3d_paint_to_vox.json` makes the shape like the Hunyuan3D-2mv template and paints it with Tencent's painter instead of projecting the pictures, so the parts no picture shows get painted too. It needs [ComfyUI-Hunyuan3DWrapper](https://github.com/kijai/ComfyUI-Hunyuan3DWrapper) installed with its rasterizer, and `hunyuan3d-dit-v2-mv_fp16.safetensors` in `models/diffusion_models`; the first run downloads the delight and paint models into `models/diffusers`.
1. Upload the pictures as for the other templates, and keep **side_faces** on `left` on both framing nodes.
2. Set **maximum_size** on **Save Voxel Model**, then **Run** and **Download**. **Save Textured Mesh** keeps the painted mesh in `output/3D`.

The painter only takes the front: **Frame Front for Delight** puts it on gray (**background** 0.8) for the delight model, the painter paints six views of the shape from it, and they are baked into the mesh's texture. **Voxelize Textured Hunyuan3D Mesh** samples that texture at **resolution** voxels along the longest axis, **samples_per_voxel** samples averaged into each, and Save Voxel Model averages those down to **maximum_size**.

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
- `placement`: `fit` crops each model to its own bounds; `grid` keeps it in place in TRELLIS's grid, every model the same size, so the frames of an animation line up.
- `fill_interior`: TRELLIS only makes the surface; on by default, fills the space it encloses with the color of the nearest surface voxel. A hole in the surface leaves that part hollow.


## Tests
`python -m unittest tests/test_vox.py`, from this directory, with `torch` installed.
