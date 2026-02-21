# NLF-Gaussian Avatar

A pipeline for training **personalized 3D avatars** from multi-view RGB images. It combines an [NLF](https://github.com/HJYblur/nlf) backbone for SMPL-X pose, and appearance prediction with a **3D Gaussian Splatting** avatar representation. You can train identity-specific avatars that can be rendered from arbitrary views.

---

## What it does

1. **Image encoding** — An NLF backbone extracts dense image features.
2. **Identity encoding** — Features are pooled over the avatar foreground to produce an identity latent vector.
3. **Avatar template** — A fixed set of Gaussians is attached to the SMPL-X mesh (e.g. \(k\) per face). Each Gaussian has a canonical position and is posed via barycentric weights on its parent triangle.
4. **Per-Gaussian features** — Backbone features are sampled at each Gaussian’s projected 2D position; 3D positions come from the precomputed posed mesh.
5. **Decoding** — A shared MLP (optionally conditioned on the identity latent) predicts per-Gaussian parameters: scale, rotation, opacity, spherical harmonics.
6. **Rendering & loss** — A differentiable 3DGS renderer produces images; training uses photometric loss [TODO: Add more losses].

The codebase uses [Lightning](https://lightning.ai/) for training and [gsplat](https://github.com/nerfstudio-project/gsplat) for rendering (GPU).

---

## Requirements

- **Python** 3.10+
- **PyTorch** 2.x
- **CUDA** for training with the differentiable renderer (CPU is supported but uses a proxy regularization loss only)

### Install dependencies
- **conda (full env with gsplat)**

If you have a Linux conda environment exported with PyTorch + gsplat:

```bash
conda env create -f environment.yml
conda activate ml
```

Adjust `environment.yml` channels/packages for your platform if needed.

---

## Repository layout

```
avatar-benchmark/
├── train.py                 # Training entrypoint
├── configs/
│   ├── nlfgs_base.yaml      # CPU / base config
│   └── nlfgs_gpu.yaml       # GPU training + rendering
├── src/
│   ├── data/
│   │   ├── datamodule.py    # Lightning DataModule
│   │   ├── datasets.py      # AvatarDataset (multi-view images)
│   │   ├── preprocess_thuman.py   # THuman 2.0 → processed views
│   │   └── preprocess_PeopleSnapshot.py
│   ├── encoder/             # NLF adapter, identity encoder, gaussian estimator
│   ├── decoder/             # Gaussian decoder MLP
│   ├── render/              # Gsplat-based renderer
│   ├── training/            # Lightning module, losses
│   └── avatar_utils/       # Config, camera, PLY, SMPL-X helpers
├── nlf/                     # Git submodule (NLF model code)
├── data/                    # Raw data & camera cache (see below)
├── processed/                # Processed per-subject images (training input)
└── models/                  # Checkpoints and templates (you provide these)
```

---

## Data preparation

Training expects a **processed** directory of per-subject folders. Each subject folder must contain multi-view images named by view.

### Expected structure

- **`data.root`** (config, default `processed`) should point to a directory containing one folder per subject.
- Inside each subject folder, images named:  
  `<subject>_front.(png|jpg|jpeg)`, `<subject>_back.*`, `<subject>_left.*`, `<subject>_right.*`
- Set **`data.num_views`** in config:
  - `1` — only `front` is loaded.
  - `4` — `front`, `back`, `left`, `right` (order in `VIEW_ORDER` in `datasets.py`).
- **`data.smplx_root`** (config, default `data/THuman_2.0_smplx_params`) should point to a directory containing one folder per subject.
- Inside each subject folder, a precomputed SMPL-X mesh named `mesh_smplx.obj`.

Example:

```
processed/
  subject_a/
    subject_a_front.png
    subject_a_back.png
    subject_a_left.png
    subject_a_right.png
  subject_b/
    ...
data/THuman_2.0_smplx_params/
  subject_a/
    mesh_smplx.obj
  subject_b/
    ...
```

### Preprocessing scripts

- **THuman 2.0**  
  Renders OBJ meshes (with textures) into the four canonical views and writes them under `processed/<identity>/` with `<identity>_front.png` etc.  
  Run from repo root (script uses paths relative to it):

  ```bash
  python -m src.data.preprocess_thuman
  ```

  Raw data is expected under `data/THuman_2.0` (see `DATA_ROOT` in the script). The script also generates camera JSONs under `data/THuman_cameras/` (e.g. `thuman_front.json`) used by the renderer.

- **[Deprecated] PeopleSnapshot**  
  Converts a PeopleSnapshot subject (cameras, video, masks, poses) into the processed layout.  
  Usage:

  ```bash
  python -m src.data.preprocess_PeopleSnapshot --root /path/to/PeopleSnapshot --subject male-3-casual --outdir processed
  ```

  You may need to adapt the output filenames to the `<subject>_<view>.png` convention if you want to use all four views with the current dataset class.

---

## Model assets

Training needs the following; paths are set in the config.

| Asset | Config key | Description |
|-------|------------|-------------|
| **NLF checkpoint** | `nlf.checkpoint_path` | TorchScript NLF model (e.g. multiperson). Default: `models/nlf_checkpoint/nlf_l_multi.torchscript`. Not included in the repo; obtain from the [NLF](https://github.com/HJYblur/nlf) project. |
| **Avatar template** | `avatar.template.path` | PLY of Gaussians on the canonical mesh. Default: `models/avatar_template.ply`. Can be generated from the canonical mesh (see below). |
| **Canonical mesh** | `avatar.template.cano_mesh_path` | SMPL-X mesh (e.g. `models/smplx/smplx_uv.obj`) used to generate or interpret the avatar template. |

### Avatar template modes

In config, **`avatar.template.mode`**:

- **`default`** — Load existing `avatar.template.path` (e.g. `models/avatar_template.ply`). Fails if the file is missing.
- **`generate`** — Build the template from the canonical mesh and save it to `avatar.template.path`.
- **`test`** — Load template and write a “test” PLY (e.g. for visualization).

Ensure the canonical mesh exists at `avatar.template.cano_mesh_path` when using `generate` or `test`.

---

## Configuration

Main config files:

- **`configs/nlfgs_base.yaml`** — CPU, 1 epoch, debug-friendly.
- **`configs/nlfgs_gpu.yaml`** — CUDA, 10 epochs, render output, loss weights.

Important sections:

- **`sys.device`** — `cpu` or `cuda` (or `cuda:0` etc.).
- **`data.root`** — Directory of processed subject folders (default `processed`).
- **`data.num_views`** — `1` or `4`.
- **`nlf.checkpoint_path`** — Path to the NLF TorchScript file.
- **`train`** — `accelerator`, `epochs`, `batch_size`, `lr`, `val_ratio`, `weight_rgb`, `weight_ssim`, etc.
- **`render.save_path`** — Where to save rendered images (e.g. `output`).

Optional: **`data.image_size`** — `[width, height]` for rendering (default `[1024, 1024]`). Should match your preprocessed image size.

---

## Running training

1. **Clone and submodules**

   ```bash
   git clone --recurse-submodules https://github.com/<your-org>/avatar-benchmark.git
   cd avatar-benchmark
   ```

2. **Prepare data**  
   Put processed per-subject images under `processed/` (or set `data.root`), with `\<subject\>_<view>.png` naming and `data.num_views` set accordingly.

3. **Prepare models**  
   - NLF: place the TorchScript checkpoint at `models/nlf_checkpoint/nlf_l_multi.torchscript` (or set `nlf.checkpoint_path`).  
   - Avatar template: have `models/avatar_template.ply` or set `avatar.template.mode: generate` and provide `models/smplx/smplx_uv.obj`.  
   - Ensure `avatar.template.cano_mesh_path` points to the SMPL-X mesh when using `generate`/`test`.

4. **Run**

   ```bash
   python train.py --config configs/nlfgs_base.yaml
   ```

   For GPU training and rendering:

   ```bash
   python train.py --config configs/nlfgs_gpu.yaml
   ```

Logs are written to `logs/train.log`. With CUDA and a valid `render.save_path`, the trainer saves rendered views under `render.save_path/<subject>/`.

---

## Debug mode

In config, set **`sys.debug: True`** to:

- Limit the dataset to a few samples.
- Optionally cache backbone features/predictions in `debug_backbone_features.pt` and `debug_backbone_preds.pt` for faster iteration.
- Save a debug input image as `debug_sample.png`.
- Export a reconstructed Gaussian PLY under `output/<subject>/` for inspection.

---

## 3DGS export / visualization

The pipeline can export predicted Gaussians to PLY (e.g. in debug mode or via `reconstruct_gaussian_avatar_as_ply` in `avatar_utils.ply_loader`). You can load and visualize these in tools such as [SuperSplat](https://superspl.at/editor) or other 3DGS viewers.

---

## License and references

- **NLF** is used as a submodule: [github.com/HJYblur/nlf](https://github.com/HJYblur/nlf). Check that project for license and checkpoint terms.
- Avatar representation and training setup follow a Gaussian-splatting-on-SMPL-X style pipeline; see config and code for details.
