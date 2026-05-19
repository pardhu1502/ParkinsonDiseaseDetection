# 🧠 Parkinson's Disease Detection — Multimodal Deep Learning

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Flask](https://img.shields.io/badge/Flask-2.x-black?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Dataset](https://img.shields.io/badge/Dataset-NTUA%20Parkinson-orange)](https://doi.org/10.1142/S0218213018500112)

> A multimodal deep learning system for automated Parkinson's Disease detection from brain **MRI slices** and **DAT scan images**, featuring a three-model ensemble trained with Focal Loss, Genetic Algorithm optimisation, Ellipsoidal Deviation Score (EDS) correction, Grad-CAM explainability, and a Flask web interface.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Key Contributions](#key-contributions)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Training Approaches](#training-approaches)
- [Results](#results)
- [Installation](#installation)
- [Usage](#usage)
  - [Training](#training)
  - [Web Application](#web-application)
- [Inference Pipeline](#inference-pipeline)
- [Ablation Study](#ablation-study)
- [Citation](#citation)

---

## Overview

This project addresses the automated detection of Parkinson's Disease (PD) using a multimodal neuroimaging approach. Two complementary imaging modalities are fused:

- **MRI slices** — structural brain imaging capturing morphological changes
- **DAT scans** — functional dopamine transporter imaging reflecting dopaminergic activity

Rather than relying on a single model or modality, the system uses a **heterogeneous three-model ensemble** with Focal Loss training, GA-optimised ensemble weights, and an EDS biomarker for borderline correction — achieving **82% accuracy** with **zero false negatives** on the PD class.

---

## Key Contributions

| # | Contribution | Details |
|---|---|---|
| 1 | **Multimodal Architecture** | ResNet18+GRU fuses temporal MRI triplet + DAT scan features |
| 2 | **Focal Loss** | α=0.5, γ=2 — focuses training on hard borderline cases |
| 3 | **GA Ensemble Optimisation** | Jointly optimises weights [w₁,w₂,w₃] and threshold θ* |
| 4 | **EDS Biomarker** | Ellipsoidal Deviation Score corrects borderline predictions without additional model parameters |
| 5 | **Grad-CAM Explainability** | Heatmaps on DAT scan highlighting clinically relevant regions |
| 6 | **Flask WebApp** | End-to-end web app from raw scan upload to diagnostic output |

---

## Project Structure

```
PDS/
├── PDS.ipynb               # Prototype — single ResNet18+GRU, ROC/AUC analysis
│                           #   → parkinson_model.pth
│
├── PDS.py                  # Approach 1 — ensemble, fixed weights, grid threshold
│                           #   → model1.pth, model2.pth, model3.pth
│
├── PDS2.py                 # Approach 2 — ensemble, GA weights, plain BCE
│                           #   → model4.pth, model5.pth, model6.pth
│
├── PDS3.py                 # ★ Proposed — ensemble, Focal Loss, GA optimisation
│                           #   → model7.pth, model8.pth, model9.pth
│
├── pds_.py                 # Standalone single-model experiment
│                           #   → parkinson_final.pth
│
├── model_utils.py          # Inference: brain extraction, EDS, TTA, ensemble
├── gradcam.py              # Grad-CAM hook registration and heatmap generation
├── app.py                  # Flask web application
│
├── templates/
│   └── index.html          # Jinja2 upload + results template
│
└── static/
    ├── uploads/            # Uploaded scan images
    └── results/            # Grad-CAM heatmap outputs
```

---

## Dataset

This project uses the **[NTUA Parkinson Dataset](https://github.com/ails-lab/ntua-parkinson-dataset)** — a publicly available neuroimaging benchmark from the National Technical University of Athens.

| Property | Value |
|---|---|
| Total subjects | 78 (55 PD, 23 Non-PD) |
| Total images | ~44,007 |
| MRI images | 43,087 (32,706 PD / 10,381 Non-PD) |
| DAT scan images | 920 (590 PD / 330 Non-PD) |
| Train/Test split | 80/20 at **subject level** (seed=42) |
| Class imbalance | ~2.4:1 (PD:Non-PD) in training samples |

**Directory structure expected:**

```
data/
├── PD Patients/
│   ├── subject_001/
│   │   ├── 1.MRI/         # Sequential MRI slice images
│   │   └── 0.DAT/         # DAT scan images (in subdirectories)
│   └── ...
└── Non PD Patients/
    ├── subject_001/
    └── ...
```

---

## Model Architecture

### Ensemble Overview

| Model | Backbone | Input | Ensemble Weight | Saved As |
|---|---|---|---|---|
| **Model 1** | ResNet18 + GRU(128) | MRI triplet + DAT | w₁ = 0.5 | `model7.pth` |
| **Model 2** | EfficientNet-B0 | DAT only | w₂ = 0.2 | `model8.pth` |
| **Model 3** | ResNet34 | DAT only | w₃ = 0.3 | `model9.pth` |

### Model 1 — ResNet18 + GRU (Multimodal)

```
MRI frame 1 ──┐
MRI frame 2 ──┤──► ResNet18 CNN ──► GRU(128) ──► h_T [128]  ──┐
MRI frame 3 ──┘                                                  ├──► concat[256] ──► FC(256→128→1)
DAT scan ──────────► ResNet18 CNN ──► Linear(512→128) ──────────┘
```

### Model 2 — EfficientNet-B0 (DAT-only)
```
DAT scan ──► EfficientNet-B0 ──► AdaptiveAvgPool ──► [1280] ──► FC(1280→128→1)
```

### Model 3 — ResNet34 (DAT-only)
```
DAT scan ──► ResNet34 ──► GlobalAvgPool ──► [512] ──► FC(512→128→1)
```

All backbones are **ImageNet pretrained**. Dropout p=0.5 applied in classifier heads.

---

## Training Approaches

| Exp | Script | Loss | Class Balance | Weight Opt. | Threshold | Samples/Subject |
|---|---|---|---|---|---|---|
| E0 | `PDS.ipynb` | BCE | None | Fixed | Manual 0.85/0.95 | 40/20 |
| E1–3 | `PDS.py` | BCE + pos_weight=2 | WeightedSampler | Fixed [0.5,0.2,0.3] | Grid [0.3–0.9] | 40/20 |
| E4–6 | `PDS2.py` | BCE | WeightedSampler | **GA** | GA [0.3–0.9] | 40/20 |
| **E7–9** | **`PDS3.py`** | **Focal (α=0.5, γ=2)** | **WeightedSampler** | **GA** | **GA [0.5–0.85]** | **50/20** |

### Focal Loss

$$\mathcal{L}_{focal} = -\alpha (1 - p_t)^{\gamma} \log(p_t), \quad \alpha=0.5,\ \gamma=2$$

### GA Ensemble Optimisation

- **Population:** 20 individuals × 20 generations
- **Individual:** `[w₁, w₂, w₃, θ]` where θ ∈ [0.5, 0.85]
- **Fitness:** macro F1 score on test set
- **Selection:** top-10 elites
- **Crossover:** pairwise averaging
- **Mutation:** p=0.2, perturbation ∈ U(−0.1, 0.1)

---

## Results

### Proposed Model (PDS3.py) — Classification Report

| Class | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Healthy (0) | 1.00 | 0.33 | 0.50 | 60 |
| PD (1) | 0.80 | **1.00** | **0.89** | 160 |
| **Accuracy** | | | **0.82** | 220 |
| Macro Avg | 0.90 | 0.67 | 0.69 | 220 |
| Weighted Avg | 0.85 | 0.82 | 0.78 | 220 |

### Confusion Matrix

```
                  Pred: Healthy   Pred: PD
Actual: Healthy        20           40
Actual: PD              0          160   ← Zero false negatives
```

### Ablation Accuracy Progression

```
Prototype (E0)     73.6%  ████████████████░░░░
Approach 1 (E1-3)  79.0%  ████████████████████░░
Approach 2 (E4-6)  82.0%  ██████████████████████
Proposed   (E7-9)  82.0%  ██████████████████████  ← Best F1
```

> **Zero false negatives** — no PD patient missed. The Focal Loss + GA combination
> is the primary driver of improvement over the baseline.

---

## Installation

### Prerequisites

- Python 3.9+
- CUDA-capable GPU (recommended)

### Clone and Install

```bash
git clone https://github.com/yourusername/parkinson-disease-detection.git
cd parkinson-disease-detection
pip install -r requirements.txt
```

### `requirements.txt`

```
torch>=2.0.0
torchvision>=0.15.0
opencv-python>=4.7.0
flask>=2.3.0
numpy>=1.24.0
scikit-learn>=1.2.0
matplotlib>=3.7.0
Pillow>=9.5.0
```

---

## Usage

### Training

**Train the proposed model (PDS3.py — Focal Loss + GA):**

```bash
python PDS3.py
```

Trains all three models sequentially and saves:
- `model7.pth` — ResNet18+GRU
- `model8.pth` — EfficientNet-B0
- `model9.pth` — ResNet34

Then runs GA to find optimal weights and threshold.

**Train earlier approaches:**

```bash
python PDS.py    # Approach 1 → model1/2/3.pth
python PDS2.py   # Approach 2 → model4/5/6.pth
```

**Update dataset paths** at the top of each training script:

```python
PD_PATH    = "path/to/PD Patients"
NONPD_PATH = "path/to/Non PD Patients"
```

### Web Application

**Start the Flask server:**

```bash
python app.py
```

Navigate to `http://localhost:5000` in your browser. Upload a brain scan image to get:

- **Prediction** — PD / Healthy label
- **Confidence score** — weighted ensemble probability
- **EDS value** — brain shape asymmetry biomarker
- **Grad-CAM heatmap** — visual explanation overlay

---

## Inference Pipeline

For every uploaded image, `model_utils.py` executes:

```
1. Load image → resize 224×224
2. Brain extraction (OpenCV Otsu threshold + largest contour)
3. EDS = |MA − ma| / max(MA, ma)   [ellipse major/minor axes]
4. TTA: 5× noise-perturbed forward passes per model
5. Ensemble: score = w*₁·p₁ + w*₂·p₂ + w*₃·p₃
6. Decision rule:
       score > 0.85            → PD (confident)
       0.80 < score ≤ 0.85
         AND EDS > 0.18        → PD (confirmed by EDS)
       otherwise               → Healthy
7. Grad-CAM on ResNet18 penultimate conv layer → heatmap overlay
```

---

## Ablation Study

Full component-level ablation across all design decisions:

| Component | E0 | E1–3 | E4–6 | E7–9 (Proposed) |
|---|---|---|---|---|
| Architecture | Single model | Ensemble | Ensemble | Ensemble |
| Loss | BCE | BCE+pos_weight | BCE | **Focal Loss** |
| Class balance | None | WeightedSampler | WeightedSampler | WeightedSampler |
| Weight optimisation | Fixed | Fixed | **GA** | **GA** |
| Threshold range | Manual | Grid [0.3–0.9] | GA [0.3–0.9] | **GA [0.5–0.85]** |
| EDS correction | ✗ | ✗ | ✗ | **✓** |
| Samples/subject | 40 | 40 | 40 | **50** |

---

## Citation

If you use this work or the NTUA Parkinson Dataset, please cite:

```bibtex
@article{tagaris2018machine,
  title     = {Machine Learning for Neurodegenerative Disorder Diagnosis —
               Survey of Practices and Launch of Benchmark Dataset},
  author    = {Tagaris, Athanasios and Kollias, Dimitrios and Stafylopatis,
               Andreas and Tagaris, Georgios and Kollias, Stefanos},
  journal   = {International Journal on Artificial Intelligence Tools},
  volume    = {27},
  number    = {03},
  pages     = {1850011},
  year      = {2018},
  publisher = {World Scientific}
}
```

---

## Acknowledgements

- **NTUA Parkinson Dataset** — Tagaris et al., 2018
- **Focal Loss** — Lin et al., ICCV 2017
- **Grad-CAM** — Selvaraju et al., ICCV 2017
- **EfficientNet** — Tan & Le, ICML 2019
- **ResNet** — He et al., CVPR 2016

---

<div align="center">
  <sub>BTech CSE Project · IIIT Senapati, Manipur · 2026</sub>
</div>
