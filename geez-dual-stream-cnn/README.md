# Dual-Stream CNN for Ge'ez Character Recognition

## Overview
Complete source code for the MSc thesis:  
**"Model-Preserving Dual-Stream Deep Learning Framework for Recognition of Degraded Ge'ez Characters"**

**Author:** Abrha Tareke Araya  
**Supervisor:** Dr. Bahailu Getachew (PhD)  
**Institution:** Mekelle University, Ethiopian Institute of Technology

---

## Features

- **Dual-Stream CNN** — Separate Base (Consonant) and Vocalic (Vowel) streams
- **Elastic Weight Consolidation (EWC)** — Continual learning to prevent catastrophic forgetting
- **Ensemble Learning** — 5-model ensemble for robust performance
- **Structured Pruning** — 45% model compression with minimal accuracy loss
- **Explainability** — Grad-CAM and SHAP for model interpretability

---

## Installation

### Clone the Repository

```bash
git clone https://github.com/abrha-tareke/geez-dual-stream-cnn.git
cd geez-dual-stream-cnn
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Quick Start

### Train with Your Own Data

```bash
python training/train.py --data_path /path/to/your/data
```

### Evaluate a Trained Model

```bash
python evaluation/evaluate.py --model weights/best_model.h5
```

---

## Repository Structure

```
geez-dual-stream-cnn/
├── README.md                 # This file
├── requirements.txt           # Python dependencies
├── Dockerfile                 # Docker container specification
├── models/
│   └── dual_stream_cnn.py    # Complete model architecture
├── data/
│   └── sample_data.py        # Synthetic data generator for testing
├── training/
│   └── train.py              # Training pipeline
├── evaluation/
│   └── evaluate.py           # Evaluation metrics and visualization
├── interpretability/
│   └── visualize.py          # Grad-CAM and SHAP implementation
└── configs/
    └── default.yaml          # Default hyperparameter configuration
```

---

## Dataset Access

⚠️ **IMPORTANT:** The full Ge'ez character dataset used in this research is **not publicly available** due to institutional and cultural heritage restrictions.

**Dataset Information:**
- 15,000 annotated character images
- 276 Ge'ez character classes
- Collected from Ethiopian Manuscript Microfilm Library (EMML), Institute of Ethiopian Studies (IES), Tigray Monasteries, and Axum Heritage Sites

**How to Request Access:**
- Contact: Ethiopian Institute of Technology, Mekelle, School of Computing
- Email: [institutional-email]
- Subject: "Ge'ez Character Dataset Access Request"

---

## Model Architecture

### Total Parameters: **4,665,948**

```
Input: 64×64×1 (Grayscale)
    ↓
Shared Convolutional Base
├── Conv2D (32) → BN → ReLU → MaxPool (2×2)
├── Conv2D (64) → BN → ReLU → MaxPool (2×2)
├── Conv2D (128) → BN → ReLU → MaxPool (2×2)
└── Conv2D (256) → BN → ReLU (NO MaxPool after Conv4!)
    ↓
    ├──────────────────────┤
    ↓                      ↓
Base Stream            Vocalic Stream
(Consonant)            (Vocalic)
Conv2D (512)           Conv2D (512)
BN → ReLU              BN → ReLU
GAP → (512)            GAP → (512)
    ↓                      ↓
    └──────────────────────┘
           ↓
    Concatenate → (1024)
           ↓
    FC1 (1024) → ReLU → Dropout (0.5)
           ↓
    FC2 (512) → ReLU → Dropout (0.5)
           ↓
    Output: 276 Classes (Softmax)
```

---

## Results Summary

| Metric | Value |
|--------|-------|
| **Recognition Accuracy** | **96.8%** |
| **Improvement over Baseline** | 7.1% - 12.6% |
| **Catastrophic Forgetting Reduction** | 15.9% → 2.4% |
| **Model Compression** | **45%** (52.3 MB → 28.8 MB) |
| **Accuracy Loss at 45% Compression** | 1.2% |

---

## License

This project is licensed under the **MIT License**.

---

## Citation

If you use this code, please cite:

```bibtex
@mastersthesis{araya2026geez,
  author = {Abrha Tareke Araya},
  title = {Model-Preserving Dual-Stream Deep Learning Framework for Recognition of Degraded Ge'ez Characters},
  school = {Mekelle University},
  year = {2026}
}
```

---

## Contact

- **Author:** Abrha Tareke Araya
- **Email:** abrha.tareke@gmail.com
- **GitHub:** https://github.com/abrha-tareke
- **Institution:** Ethiopian Institute of Technology, Mekelle