# FGC-DETR

ITC-Asia: Factor Graph–Optimized Cross-Scale Detector for Wafer Defect Inspection
- FGC-DETR: https://github.com/mindednessr/FGC-DETR

<p align="center">
  <img src="figures/fgc_arch.png" alt="FGC-DETR Architecture" width="95%" />
</p>

FGC-DETR is a wafer-defect detection repository inspired by RT-DETRv4. It encapsulates end-to-end training and inference pipelines, a flexible configuration system, and a suite of utility scripts intended for both research reproducibility and industrial deployment.

This codebase is meticulously engineered to provide a modular and reproducible foundation—featuring multiple backbone choices, configurable training schedules, and dataset adapters to streamline experimentation and productionization.

Contents
- Core training and evaluation scripts: `train.py`, `test.py`.
- Configuration system: `configs/` (YAML-driven configurations for models, datasets, and training regimes).
- Modular components: `engine/`, `core/`, `data/`, `misc/` house the model implementations, data loaders, transforms, and utility functions.
- Visualization and diagnostics: `tools/visualization` and `plot.py` assist with monitoring and qualitative inspection of results.

Key Features
- An end-to-end detector specifically tailored for wafer-defect inspection under extreme scale variance.
- Cross-scale feature alignment to harmonize multi-resolution cues.
- Defect edge enhancement modules to improve localization and delineation of irregular patterns.

Note: This repository contains the principal implementation elements for the FGC-DETR paper and is built upon the RT-DETR framework. While not a line-by-line reproduction, the core algorithms and components faithfully reflect the methodology described in our manuscript.

Quickstart

Repository references:
- RT-DETRv4: https://github.com/RT-DETRs/RT-DETRv4
- YOLOs (Ultralytics): https://github.com/ultralytics/ultralytics

1. Environment setup

We recommend Python 3.8+ and a compatible CUDA/cuDNN stack for GPU acceleration. Install dependencies listed in `requirements.txt` inside an isolated virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

Conda alternative:

```bash
conda create -n fgc-detr python=3.9 -y
conda activate fgc-detr
pip install -r requirements.txt
```

2. Dataset configuration

The codebase supports COCO-style datasets as well as common bespoke detection datasets. See `configs/dataset/` for templates (e.g., `coco_detection.yml`, `custom_detection.yml`) and modify file paths and class definitions accordingly.

Example: COCO2017 (wafer_datasets: WM-811k and MixedWM38)

1. Download COCO2017 from OpenDataLab or the official COCO site.
2. Edit the dataset paths in `configs/dataset/coco_detection.yml`:

```yaml
train_dataloader:
    img_folder: /data/COCO2017/train2017/
    ann_file: /data/COCO2017/annotations/instances_train2017.json
val_dataloader:
    img_folder: /data/COCO2017/val2017/
    ann_file: /data/COCO2017/annotations/instances_val2017.json
```

For custom datasets, organize images and annotations in COCO format and set `remap_mscoco_category: False` if your category IDs differ from MS COCO.

3. Training example

Launch a training run using a chosen configuration:

```bash
python train.py --config configs/fgc_detr_wafer.yml --work-dir ./logs/exp1
```

Common arguments
- `--config`: path to the YAML configuration file.
- `--work-dir`: output directory for logs and checkpoints.

4. Evaluation and inference

Evaluate a checkpoint:

```bash
python test.py --config configs/fgc_detr_wafer.yml --checkpoint path/to/checkpoint.pth --eval bbox
```

Run single-image inference (see `tools/inference/demo.py`):

```bash
python tools/inference/demo.py --config configs/fgc_detr_wafer.yml --checkpoint path/to/checkpoint.pth --image data/sample.jpg
```

Configuration overview

- `configs/`: the primary configuration directory, organized by base templates, datasets, and model families (e.g., `base/`, `dataset/`, `deim/`, `rtv2/`, `rtv4/`).
- Each YAML config encapsulates model architecture, optimizer settings, learning-rate schedules, data augmentation pipelines, and dataset paths.

Logs & checkpoints

- Training logs and model checkpoints are saved under `logs/` or a directory specified with `--work-dir`.
- Checkpoints use PyTorch `.pth` format and can be loaded with `test.py` for evaluation or `tools/inference` for deployment.

Contributing

Contributions are welcome. When reporting bugs or reproducibility issues, please provide the command used, the configuration file, and environment details. For new model or dataset adapters, add configurations under `configs/` and corresponding data loaders or transforms under `data/`.

Acknowledgements

This project integrates ideas and implementations from multiple open-source repositories; we gratefully acknowledge those contributions.

License

This project is governed by the LICENSE file at the repository root. Please review it prior to reuse.

Contact

For assistance or collaboration inquiries, open an issue or contact the repository maintainers.
