# FGC-DETR

**ITC-Asia — Factor Graph–Optimized Cross-Scale Detector for Wafer Defect Inspection**

<p align="center">
  <img src="figures/fgc_arch.png" alt="FGC-DETR Architecture" width="95%" />
</p>

Overview
--------

FGC-DETR is a research-grade and production-ready codebase for wafer-defect detection. Built on RT-DETRv4 concepts, it unifies robust model implementations, experiment configuration, and utility tooling to accelerate reproducible studies and deployment.
- FGC-DETR: https://github.com/mindednessr/FGC-DETR

Table of Contents
-----------------

- [FGC-DETR](#fgc-detr)
  - [Overview](#overview)
  - [Table of Contents](#table-of-contents)
  - [Key Features](#key-features)
  - [Quickstart](#quickstart)
  - [Dataset Configuration](#dataset-configuration)
  - [Training](#training)
  - [Evaluation \& Inference](#evaluation--inference)
  - [Configuration Overview](#configuration-overview)
  - [Logs \& Checkpoints](#logs--checkpoints)
  - [Contributing](#contributing)
  - [Acknowledgements](#acknowledgements)
  - [License](#license)
  - [Contact](#contact)

Key Features
------------

- End-to-end detector tailored for wafer-defect inspection with extreme scale variance.
- Cross-scale feature alignment for consistent multi-resolution representation.
- Defect edge enhancement modules to improve localization and boundary delineation.

Quickstart
----------

Repository references

- RT-DETRv4: https://github.com/RT-DETRs/RT-DETRv4
- Ultralytics (YOLO family): https://github.com/ultralytics/ultralytics

Environment (recommended)

```bash
python -m venv .venv
.venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

Conda alternative

```bash
conda create -n fgc-detr python=3.9 -y
conda activate fgc-detr
pip install -r requirements.txt
```

Dataset Configuration
---------------------

The codebase expects COCO-style annotations. See `configs/dataset/` for templates (e.g., `coco_detection.yml`, `custom_detection.yml`). Update image folders and annotation paths accordingly.

Example (COCO2017):

```yaml
train_dataloader:
  img_folder: /data/COCO2017/train2017/
  ann_file: /data/COCO2017/annotations/instances_train2017.json
val_dataloader:
  img_folder: /data/COCO2017/val2017/
  ann_file: /data/COCO2017/annotations/instances_val2017.json
```

For bespoke datasets, provide COCO-format annotations and set `remap_mscoco_category: False` if your class IDs differ from MS COCO.

Training
--------

Start training with a chosen configuration:

```bash
python train.py --config configs/fgc_detr_wafer.yml --work-dir ./logs/exp1
```

Common arguments

- `--config`: YAML configuration file path.
- `--work-dir`: directory for logs and checkpoints.

Evaluation & Inference
----------------------

Evaluate a checkpoint (example):

```bash
python test.py --config configs/fgc_detr_wafer.yml --checkpoint path/to/checkpoint.pth --eval bbox
```

Run single-image inference (see `tools/inference/demo.py`):

```bash
python tools/inference/demo.py --config configs/fgc_detr_wafer.yml --checkpoint path/to/checkpoint.pth --image data/sample.jpg
```

Configuration Overview
----------------------

- `configs/`: primary configuration directory. Organized into `base/`, `dataset/`, and model family subfolders (e.g., `deim/`, `rtv2/`, `rtv4/`).
- Each YAML file defines model architecture, optimizer, LR schedule, augmentations, and dataset paths.

Logs & Checkpoints
------------------

- Training outputs (logs and checkpoints) are saved to `logs/` or a directory specified via `--work-dir`.
- Checkpoints are standard PyTorch `.pth` files; loadable via `test.py` and `tools/inference` utilities.

Contributing
------------

Contributions are welcome. When opening issues or PRs, include the command used, full config file, and environment details to facilitate reproducibility. For new adapters, add configurations under `configs/` and corresponding loaders/transforms under `data/`.

Acknowledgements
----------------

This project builds upon and integrates multiple open-source efforts; we gratefully acknowledge upstream contributors.

License
-------

See the `LICENSE` file at the repository root for licensing terms.

Contact
-------

For questions or collaboration, open an issue or contact the maintainers.
