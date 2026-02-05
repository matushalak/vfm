# ABC-VFM: E(3)-equivariant Variational Flow Matching for Molecular Generation

This repository contains the implementation of the **ABC-Transformer**, an $E(3)$-equivariant graph transformer designed for the joint generation of discrete molecular structures (Atoms and Bonds) and continuous 3D geometries (Coordinates). The model is trained using the **Variational Flow Matching (VFM)** framework.

This work was carried out as a Graph Generative Modeling project for the Machine Learning for Graphs MSc. Artificial Intelligence course @ VU Amsterdam.

## 🌟 Key Features
* **ABC-Transformer Architecture**: A novel backbone handling four concurrent data streams: Atoms ($X$), Bonds ($E$), Coordinates ($C$), and Global features ($y$).
* **Joint Molecular Generation**: Simultaneously generates chemically valid discrete graphs and geometrically consistent 3D atomic positions.
* **$E(3)$ Equivariance**: Guaranteed symmetry preservation for molecular rotations, translations, and reflections via equivariant coordinate updates.
* **VFM Framework**: Implementation of Mean-Field Variational Flow Matching and CatFlow for stable generative modeling of categorical and continuous data.

## 📁 Repository Structure
```bash
.
├── architecture.py          # Core ABC-Transformer and Equivariant blocks
├── vanilla_transformer.py   # Baseline DiGress-style Graph Transformer
├── vfm.py                   # Variational Flow Matching logic and loss functions
├── experiment.py            # Main entry point for training and evaluation
├── pretrained.py            # Main entry point for using pre-trained models (for sampling & evaluation)
├── molecule_data.py         # QM9 dataset loading and preprocessing, generated molecule evaluation
├── my_utils.py              # Masking, placeholders, and geometric utilities
├── config.py                # Hyperparameter and model configurations
├── runs/                    # Training logs and saved model checkpoints & samples
├── examples/                # Representative examples of discrete / 3d sampled molecules with our approaches
└── env_gpu.yml              # Conda environment for CUDA-enabled training
└── env_mac.yml              # Conda environment for Metal-enabled training on macOS
├── tests.py                 # Miscellaneous tests
├── fm-vs-vfm.ipynb          # Notebook contrasting FM vs VFM on toy datasets (half-moons, checkerboard)
├── molecules.ipynb          # Notebook for exploration of QM9 and ZINC molecular datasets

## 🚀 Getting Started

### 1. Installation
Clone the repository and set up the environment using the provided YAML files (optimized for either GPU or MacOS/M-series):

```bash
conda env create -f env_gpu.yml
conda activate vfm

### 2. Training & Evaluation
To train and evaluate the Vanilla DiGress-Transformer on the QM9 dataset for discrete molecular generation:
```bash
python experiment.py --epochs 1000 --bs 1024 --lr 1e-3

To train and evaluate the ABC-Transformer on the QM9 dataset for joint molecular generation:
```bash
python experiment.py --epochs 1000 --bs 1024 --lr 1e-3 --keep_pos True

### 3. Sampling
To sample new molecules using pre-trained models use:
```bash
python pretrained.py

## 📜 Citation
If you use this code in your research, please cite:
@article{halak2026abcvfm,
  title={E(3)-equivariant Variational Flow Matching for Molecular Generation},
  author={Halák, Matúš},
  year={2026},
  school={VU Amsterdam}
}