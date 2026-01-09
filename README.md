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

### 2. CNN 베이스라인 학습
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

### 3. CSANN 베이스라인 학습
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

### 4. GQHAN 베이스라인 학습
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

### 5. QSAN 베이스라인 학습
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

## 주요 옵션

### 공통 옵션
- `--dataset-choice`: 데이터셋 선택 (`mnist`, `fmnist`, `cifar10`, `pcam`)
- `--train-count`, `--val-count`, `--test-count`: 학습/검증/테스트 샘플 수
- `--batch-size`: 배치 크기
- `--epochs`: 학습 에폭 수
- `--learning-rate`: 학습률
- `--device`: 학습 디바이스 (`cpu`, `cuda:0`)
- `--seed`: 랜덤 시드 (재현성)
- `--early-stop`: 조기 종료 활성화

### QSANN 전용 옵션
- `--num-qubits`: 큐비트 수
- `--vqc-layers`: VQC 레이어 수
- `--reuploading`: 데이터 인코딩 반복 횟수
- `--attn-layers`: 어텐션 레이어 수
- `--classification-task`: `binary` 또는 `multi` 분류

### 데이터 필터링
- `--dataset-labels`: 특정 레이블만 사용 (예: `--dataset-labels 0 1`)
- `--samples-per-label`: 레이블당 샘플 수 제한

## 출력
- 학습 로그: `results/logs/` 디렉토리
- 모델 체크포인트: `results/models/` 디렉토리
- 각 에폭마다 Train/Val/Test 정확도, AUROC, Precision, Recall, F1-score 출력
