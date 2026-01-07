# Multimodal Contrastive Learning for Automated Alzheimer's Disease Prediction

[![Paper](https://img.shields.io/badge/ICCV%202023-Workshop-blue)](https://openaccess.thecvf.com/content/ICCV2023W/CVAMD/papers/Huang_Multimodal_Contrastive_Learning_and_Tabular_Attention_for_Automated_Alzheimers_Disease_ICCVW_2023_paper.pdf)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

This repository contains the implementation of our **ICCV 2023 Workshop** paper on multimodal contrastive learning for Alzheimer's disease prediction using MRI scans and clinical tabular data from the ADNI dataset.

## Overview

We propose a multimodal contrastive learning framework that learns joint representations from:
- **3D MRI Brain Volumes** — Encoded using a 3D ResNet architecture
- **Clinical Tabular Data** — Encoded using TabTransformer with attention mechanisms

The model learns to align representations across modalities using a CLIP-style contrastive loss, enabling effective prediction of Alzheimer's disease progression.

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/iccv23.git
cd iccv23

# Install dependencies
pip install -r requirements.txt
```

### Requirements
- Python 3.8+
- PyTorch 1.10+
- CUDA-capable GPU (recommended)

## Data Setup

1. **ADNI Data**: Download the ADNI dataset from [ADNI website](https://adni.loni.usc.edu/)
2. **Metadata CSV**: Download from [Google Drive](https://drive.google.com/file/d/1N77WWToQJ_PJ7MNwg1VrxHoCNYVlzMod/view?usp=sharing)
3. Place data in the following structure:
   ```
   data/
   ├── adni/
   │   ├── ADNI/          # NIfTI files organized by patient
   │   └── metadata.csv
   └── tadpole_challenge/
       └── TADPOLE_D1_D2.csv
   ```

## Usage

### Standard Training

```bash
# Train all modalities
python scripts/train.py --epochs 64 --batch_size 8

# Train specific modality
python scripts/train.py --modality dx --epochs 64
```

### Cross-Validation Training

```bash
# 5-fold cross-validation with early stopping
python scripts/train.py --cross_validation --n_folds 5 --early_stopping_patience 5

# Custom configuration
python scripts/train.py \
    --cross_validation \
    --n_folds 5 \
    --epochs 100 \
    --batch_size 8 \
    --lr 1e-3 \
    --output_dir checkpoints
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--epochs` | 64 | Maximum training epochs |
| `--batch_size` | 8 | Batch size |
| `--lr` | 1e-3 | Learning rate |
| `--weight_decay` | 1e-3 | Weight decay |
| `--cross_validation` | False | Enable K-fold cross-validation |
| `--n_folds` | 5 | Number of CV folds |
| `--early_stopping_patience` | 5 | Early stopping patience |
| `--modality` | all | Modality to train: `dx`, `cog`, `vol`, `pet`, `bio`, `demo`, or `all` |
| `--output_dir` | checkpoints | Output directory for models and results |

## Project Structure

```
iccv23/
├── dataset/
│   ├── dataset.py         # TemporalDataset with cross-validation support
│   └── utils.py            # Data loading and preprocessing utilities
├── modelling/
│   ├── contrastive.py      # ContrastiveModel with global/local loss
│   ├── encoders.py         # Image, Volume, Tabular, and TabTransformer encoders
│   └── losses.py           # CLIP and InfoNCE losses
├── scripts/
│   └── train.py            # Training script with CV support
├── validation.py           # Cross-validation infrastructure
├── logger/                 # Logging utilities
├── testing/                # Unit tests
├── requirements.txt
└── README.md
```

## Modality Groups

The model supports multiple clinical data modalities:

| Modality | Features |
|----------|----------|
| `dx` | Diagnosis, ADAS13, Ventricles, ExamDate |
| `cog` | CDRSB, ADAS11, MMSE, RAVLT Immediate |
| `vol` | Hippocampus, WholeBrain, Entorhinal, MidTemp |
| `pet` | FDG, AV45 |
| `bio` | ABETA, TAU, PTAU |
| `demo` | APOE4, Age |

## Citation

If you find this work useful, please cite our paper:

```bibtex
@InProceedings{Huang_2023_ICCV,
    author    = {Huang, Weichen and Xing, Xixi and Gao, Junxin and Su, Yuhan and Venkataramanan, Shiv and Lam, Bryan and Shen, Li and Nie, Jingshan and Liu, Yue},
    title     = {Multimodal Contrastive Learning and Tabular Attention for Automated Alzheimer's Disease Prediction},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV) Workshops},
    month     = {October},
    year      = {2023},
    pages     = {2589-2597}
}
```


## Acknowledgments

Data used in preparation of this article were obtained from the Alzheimer's Disease Neuroimaging Initiative (ADNI) database.
