# BD-ResTransUNet: A Boundary-Driven Dual-Stream Framework for SLE Lesion Segmentation

Official implementation of the paper: **"BD-ResTransUNet: A Boundary-Driven Dual-Stream Framework with Geometric-Adaptive Attention for Robust Segmentation of Systemic Lupus Erythematosus Lesions"**

---

## 🔬 Architectural Innovations
Our framework addresses the inherent contradiction between semantic abstraction and structural preservation in complex medical imaging via three core modules:

1. **Explicit Boundary Stream (EBS):** Anchors high-frequency anatomical margins via hard inductive bias.
2. **Geometric-Adaptive Hybrid Deformable Attention (HDA):** Dynamically reconfigures receptive fields to capture non-rigid lesion topologies.
3. **Multi-Scale Synergistic Fusion (MSF):** A semantic gateway for heterogeneous feature alignment and purification.

## Pretrained Weights

Due to GitHub file size limitations, the pretrained model weights are hosted externally.

You can download all trained weights from the following link:
https://drive.google.com/drive/folders/1WikwHWSnkOD1dK4vNTh_zZSunm2YVC9u?usp=sharing

## 📁 Repository Structure
```text
├── models/
│   ├── abstract_layers.py   # HDA and MSF operator implementations
│   ├── core_framework.py    # Dual-stream interaction logic
│   └── constants.py         # Hyperparameter configurations
├── engines/
│   ├── execution_engine.py  # Orchestrator for training & validation
│   └── metrics_factory.py   # Clinical metrics (HD95, Dice, etc.)
├── data_loader/             # Abstracted data pipeline
└── requirements.txt         # Environment specifications
