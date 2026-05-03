# NLF-GS

A model for training **personalized 3D avatars** from sparse multi-view images. 
<!-- A **ResNet50-FPN** backbone extracts image features; Gaussians live on a fixed **SMPL-X**-anchored template and are decoded into 3D Gaussian Splatting parameters, then rendered with **[gsplat](https://github.com/nerfstudio-project/gsplat)**. Training is driven by **[PyTorch Lightning](https://lightning.ai/)** with photometric and regularization losses. -->
---

## Requirements

NVIDIA GPUs are required for this project. We recommend using anaconda to manage the python environments.

```bash
  conda create -n ml python=3.10
  conda activate ml
  # Install PyTorch appropriate for your system
  pip install torch torchvision torchaudio
  # Then install other packages
  conda env update -f environment.yml
```

## Data Setup

### Download THuman2.0 DataSet
Please follow the[original repo](https://github.com/ytrock/THuman2.0-Dataset) to ask for permission to download the raw obj scans and estimated smplx parameters following their instructions. We only use the first 525 subjects in this experiment. After downloading, put the scaned obj files under `./data/Thuman_2.0` and the smplx parameters under `./data/THuman_2.0_smplx`.

The final `data` folder should look like:
```
data/
├── Thuman_2.0/
│   ├── 0001/
│   │   └── 0001.obj
│   │   └── material0.jpeg
│   │   └── material0.mtl
│   ├── 0002/
│   │   └── ...
│   └── ...
└── THuman_2.0_smplx/
    ├── 0001/
    │   └── smplx_param.pkl
    │   └── mesh_smplx.obj
    ├── 0002/
    │   └── ...
    └── ...

```

If there is wierd `_.*` files, use the `scripts/clean_dot_underscore.sh` to remove them.
```
chmox +x scripts/clean_dot_underscore.sh
./scripts/clean_dot_underscore.sh # List all the files
./scripts/clean_dot_underscore.sh --delete # Delete all the files
```

### Download smplx models
Please create a `smplx` folder under `models` and download the smplx models from [smplx web](https://smpl-x.is.tue.mpg.de/).

The final `models` folder should look like:
```
models/
├── smplx/
│   ├── SMPLX_FEMALE.npz
│   ├── SMPLX_MALE.npz
│   └── SMPLX_NEUTRAL.npz
│   ├── SMPLX_FEMALE.pkl
│   ├── SMPLX_MALE.pkl
│   └── SMPLX_NEUTRAL.pkl
│   └── smplx_uv.obj
│   └── smplx_uv.png
└── ...
```

### Data Process

Before the actual training, we need to render ground truth image from the scanned obj files. You can achieve this by running
```
python src/data/preprocess_thuman.py
```

All the preprocessed data will be saved under `processed` automatically, with the structure looks like:

```text
processed/
├── 0001/
│   ├── 0001_0.png
│   ├── 0001_0_mask.png
│   ├── 0001_15.png
│   ├── 0001_15_mask.png
│   │   # … every 15° through …
│   ├── 0001_345.png
│   ├── 0001_345_mask.png
│   └── smplx_param.pkl
├── 0002/
│   └── ...
└── 0525/
    ├── 0525_0.png
    ├── 0525_0_mask.png
    │   # … 24 azimuth views (0° = front, 180° = back) …
    └── smplx_param.pkl
```

Camera intrinsics / extrinsics for Gaussian rendering are written under `data/THuman_cameras/` as `thuman_0.json` … `thuman_345.json` (run `preprocess_thuman` or call `generate_camera_mapping()` to create them).


## Code Run
### Train 

```bash
python train.py
```

### Inference

For generating an avatar for a unseen subject, you should run the following command. 

```bash
python inference.py
```
### Evaluation
For NLF-GS:
```bash
python src/evaluation/compute_metrics.py --target-root output_gt   --preds-root output --no-mask
```
For GHG:
```bash
python src/evaluation/compute_metrics.py --target-root GHG_dataset/gt   --preds-root GHG_dataset/pred --no-mask
```