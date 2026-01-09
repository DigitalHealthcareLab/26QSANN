# QSANN Usage Guide

## Overview
Training and evaluation code for QSANN (Quantum Self-Attention Neural Network) and baseline models.

## Model Types
- **QSANN** (`main.py`)
- **CNN** (`baselines/CNN.py`)
- **CSANN** (`baselines/CSANN.py`)
- **GQHAN** (`baselines/GQHAN.py`)
- **QSAN** (`baselines/QSAN.py`)

## Basic Usage

### 1. Train QSANN
```bash
python main.py \
  --dataset-choice mnist \
  --classification-task binary \
  --train-count 1000 \
  --val-count 100 \
  --test-count 100 \
  --image-size 28 \
  --patch-size 4 \
  --num-qubits 8 \
  --vqc-layers 1 \
  --reuploading 3 \
  --epochs 300 \
  --batch-size 32 \
  --learning-rate 0.05 \
  --device cuda:0
```

### 2. Train CNN Baseline
```bash
python baselines/CNN.py \
  --dataset-choice cifar10 \
  --train-count 1000 \
  --val-count 100 \
  --test-count 100 \
  --image-size 32 \
  --batch-size 16 \
  --epochs 30 \
  --learning-rate 0.001 \
  --device cuda:0
```

### 3. Train CSANN Baseline
```bash
python baselines/CSANN.py \
  --dataset-choice mnist \
  --train-count 1000 \
  --val-count 100 \
  --test-count 100 \
  --image-size 28 \
  --patch-size 4 \
  --attn-layers 1 \
  --hidden-dim 64 \
  --epochs 30 \
  --device cuda:0
```

### 4. Train GQHAN Baseline
```bash
python baselines/GQHAN.py \
  --dataset-choice mnist \
  --train-count 1000 \
  --val-count 100 \
  --test-count 100 \
  --pca-dim 8 \
  --epochs 30 \
  --device cuda:0
```

### 5. Train QSAN Baseline
```bash
python baselines/QSAN.py \
  --dataset-choice mnist \
  --train-count 1000 \
  --val-count 100 \
  --test-count 100 \
  --image-size 28 \
  --patch-size 4 \
  --n-qubits 4 \
  --epochs 30 \
  --device cuda:0
```

## Key Arguments

### Common Options
- `--dataset-choice`: Dataset selection (`mnist`, `fmnist`, `cifar10`, `pcam`)
- `--train-count`, `--val-count`, `--test-count`: Number of train/validation/test samples
- `--batch-size`: Batch size
- `--epochs`: Number of training epochs
- `--learning-rate`: Learning rate
- `--device`: Device for training (`cpu`, `cuda:0`)
- `--seed`: Random seed for reproducibility
- `--early-stop`: Enable early stopping

### QSANN-Specific Options
- `--num-qubits`: Number of qubits
- `--vqc-layers`: Number of VQC layers
- `--reuploading`: Number of data encoding repetitions
- `--attn-layers`: Number of attention layers
- `--classification-task`: Classification task (`binary` or `multi`)

### Data Filtering
- `--dataset-labels`: Use only specific labels (e.g., `--dataset-labels 0 1`)
- `--samples-per-label`: Limit samples per label

## Output
- Training logs: `results/logs/` directory
- Model checkpoints: `results/models/` directory
- Outputs Train/Val/Test accuracy, AUROC, Precision, Recall, F1-score per epoch
