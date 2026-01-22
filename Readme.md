# Paired-Sampling Contrastive Framework for Face Attack Detection

[![Paper](https://img.shields.io/badge/arXiv-2508.14980-b31b1b.svg)](https://arxiv.org/abs/2508.14980)
[![ICCV Workshop](https://img.shields.io/badge/ICCV%202025-Workshop-blue.svg)](https://sites.google.com/view/face-anti-spoofing-challenge)

Official implementation of **"Paired-Sampling Contrastive Framework for Joint Physical-Digital Face Attack Detection"** (ICCV 2025 Workshop).

## Overview

This framework unifies **Presentation Attack Detection (PAD)** and **Deepfake Detection (DFD)** under a single model architecture. Key features:

- **Paired sampling**: Matches genuine and attack selfies by identity for robust training
- **Asymmetric augmentation**: Applies augmentations only to genuine samples
- **Contrastive learning**: Combined focal loss and supervised contrastive loss
- **Lightweight**: Only 4.46 GFLOPs with ConvNeXt-v2-Tiny backbone
- **Fast training**: Under 1 hour on 2 GPUs

**Results**: ACER of **2.10%** on the 6th Face Anti-Spoofing Challenge benchmark.

## Installation

### Using Docker (Recommended)

```bash
cd docker
docker build -t deepfake-detector:latest .
```

### Manual Installation

```bash
pip install -r requirements.txt
```

## Quick Start

### Training

```bash
# Single GPU
python tools/train.py --config configs/default.py

# Multi-GPU (auto-detects available GPUs)
python tools/train.py --config configs/default.py

# Specify GPUs
python tools/train.py --config configs/default.py --gpus 0,1
```

### Evaluation

```bash
python tools/eval.py \
    --checkpoint weights/pt/best.pth \
    --protocol dataset/Protocol-val-test.txt \
    --root-dir /path/to/data \
    --bbox-csv dataset/val_bbox.csv \
    --output submission.txt
```

## Project Structure

```
├── configs/
│   └── default.py          # Training configuration
├── src/
│   ├── dataset/            # Data loading and augmentation
│   ├── models/             # Model architectures
│   ├── loss_utils.py       # Loss functions (Focal, SupCon)
│   ├── training_procedure.py
│   └── stat_keeper.py      # Metrics computation
├── tools/
│   ├── train.py            # Training script
│   └── eval.py             # Evaluation script
├── prepare_data/           # Data preparation utilities
│   ├── extract_embeddings.py
│   ├── crop_faces_v2.py
│   ├── filter_train.py
│   └── bbox_to_data.py
├── docker/
│   └── Dockerfile
└── dataset/                # Protocol files (not included)
```

## Data Preparation

### 1. Extract Face Bounding Boxes

```bash
python prepare_data/crop_faces_v2.py \
    --src /path/to/raw/images \
    --dst /path/to/crops \
    --csv-name bboxes.csv
```

### 2. Extract Face Embeddings

```bash
python prepare_data/extract_embeddings.py \
    --src /path/to/images \
    --dst /path/to/embeddings
```

### 3. Create Paired Training Data

```bash
python prepare_data/filter_train.py \
    --data-dir /path/to/data \
    --embeddings-dir /path/to/embeddings \
    --clean-df-path dataset/train_clean.csv \
    --sim-thr 0.90 \
    --output-csv dataset/train_matched.csv
```

## Configuration

Key parameters in `configs/default.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model.model_name` | `convnextv2_tiny` | Backbone architecture |
| `loss.type` | `binary_focal` | Loss function |
| `loss.supcon_weight` | `0.306` | Weight for contrastive loss |
| `cutmix_prob` | `0.3` | CutMix probability |
| `trainer.epochs` | `20` | Number of training epochs |

## Docker Usage

### Training with Docker

```bash
docker run --gpus all -it --rm \
    -v $(pwd):/workspace \
    -v /path/to/dataset:/dataset \
    -v /path/to/data:/data \
    -w /workspace \
    --shm-size=16gb \
    deepfake-detector:latest \
    tools/train.py --config configs/default.py
```

### Evaluation with Docker

```bash
docker run --gpus all -it --rm \
    -v $(pwd):/workspace \
    --shm-size=16gb \
    deepfake-detector:latest \
    tools/eval.py --checkpoint weights/pt/best.pth --protocol /workspace/dataset/Protocol-test.txt
```

## Citation

```bibtex
@inproceedings{balykin2025paired,
    title={Paired-Sampling Contrastive Framework for Joint Physical-Digital Face Attack Detection},
    author={Balykin, Andrei and Ganiev, Anvar and Kondranin, Denis and Polevoda, Kirill and Liudkevich, Nikolai and Petrov, Artem},
    booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops},
    year={2025}
}
```

## License

This project is released under the MIT License.

## Acknowledgements

- [UniAttackData](https://github.com/ZitongYu/UniAttackData) for the benchmark dataset
- [timm](https://github.com/huggingface/pytorch-image-models) for pretrained backbones
- [facenet-pytorch](https://github.com/timesler/facenet-pytorch) for face detection
