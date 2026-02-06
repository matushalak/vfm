# ABC-VFM: E(3)-equivariant Variational Flow Matching for Molecular Generation

This repository contains the implementation of the **ABC-Transformer**, an $E(3)$-equivariant graph transformer designed for the joint generation of discrete molecular structures (Atoms and Bonds) and continuous 3D geometries (Coordinates). The model is trained using the **Variational Flow Matching (VFM)** framework.

This work was carried out as a Graph Generative Modeling project for the Machine Learning for Graphs MSc. Artificial Intelligence course @ VU Amsterdam.

## 🌟 Key Features
* **ABC-Transformer Architecture**: A novel backbone handling four concurrent data streams: Atoms ($X$), Bonds ($E$), Coordinates ($C$), and Global features ($y$).
* **Joint Molecular Generation**: Simultaneously generates chemically valid discrete graphs and geometrically consistent 3D atomic positions.
* **$E(3)$ Equivariance**: Guaranteed symmetry preservation for molecular rotations, translations, and reflections via equivariant coordinate updates.
* **VFM Framework**: Implementation of Mean-Field Variational Flow Matching and CatFlow for stable generative modeling of categorical and continuous data.

## 📁 Repository Structure
```
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
```

## 🚀 Getting Started
### 1. Installation
Clone the repository and set up the environment using the provided YAML files (optimized for either GPU or MacOS/M-series):

```
conda env create -f env_gpu.yml
conda activate vfm
```
### 2. Training & Evaluation
To train and evaluate the Vanilla DiGress-Transformer on the QM9 dataset for discrete molecular generation:
```
python experiment.py --epochs 1000 --bs 1024 --lr 1e-3
```
To train and evaluate the ABC-Transformer on the QM9 dataset for joint molecular generation:
```
python experiment.py --epochs 1000 --bs 1024 --lr 1e-3 --keep_pos True
```
### 3. Sampling
To sample new molecules using pre-trained models use:
```
python pretrained.py
```

## 🧠 Architecture Details
The **ABC-Transformer** is a specialized graph transformer designed to joint-model discrete molecular features and continuous 3D coordinates while strictly adhering to $E(3)$-equivariance. The architecture processes four concurrent data streams—**A**toms ($X$), **B**onds ($E$), **C**oordinates ($C$), and global features ($y$)—through a series of equivariant blocks.

### 1. E(3)-Equivariant Coordinate Update
The core innovation is the `E3NodeEdgeCoordBlock`, which updates atomic positions using a multi-head attention-modulated rule that ensures the model's outputs remain consistent regardless of the molecule's rotation or translation in space. The coordinate update $\Delta c$ for a node $i$ is defined as:

$$\Delta c_{i} = \sum_{h=1}^H \eta_t^{(h)} \sum_{j=1}^N (a_{ij}^{(h)} s_{ij}^{(h)} \hat{r}_{ij})$$

* **$\eta_t^{(h)}$**: A time-dependent learning rate for each head $h$, derived from the global features $y$.
* **$a_{ij}^{(h)}$**: Scaled dot-product attention scores that determine the importance of neighbor $j$ in updating the position of node $i$.
* **$s_{ij}^{(h)}$**: Scalar gates derived from the concatenation of node, edge, coordinate, and global features that determine the magnitude and sign of the spatial shift.
* **$\hat{r}_{ij}$**: Unit-length relative coordinate vectors $\frac{c_i - c_j}{||c_i - c_j||_2}$ providing the equivariant direction for the update.

### 2. Feature Stream Updates
Each transformer layer updates the discrete and global streams to maintain a rich representation of the molecular graph:
* **Node ($X$) and Edge ($E$) Streams**: These features are updated via multi-head self-attention. Global context is integrated into these streams using **FiLM** (Feature Wise Linear Modulation) layers, which apply affine transformations based on the global vector $y$.
* **Global Stream ($y$)**: Global graph features are updated using **Principal Neighborhood Aggregation (PNA)**, which aggregates information across all nodes and edges to provide a comprehensive representation of the entire molecule.
* **Pairwise Enrichment**: Before computing the scalar gates, the model expands $E(3)$-invariant pairwise distances $d_{ij}$ using a **Radial Basis Function (RBF)**, which is then processed through an MLP to inform $s_{ij}^{(h)}$.


### 3. Symmetry and Preservation
* **Permutation Equivariance**: Maintained through the use of neighborhood aggregation and self-attention, ensuring the model is invariant to the ordering of atoms in the input.
* **Coordinate Stability**: While node, edge, and global features are "lifted" into higher-dimensional latent spaces, coordinates remain in $\mathbb{R}^3$ throughout the network to preserve their geometric meaning and equivariance.

## 📜 Citation
If you use this code in your research, please cite:
@article{halak2026abcvfm,
  title={E(3)-equivariant Variational Flow Matching for Molecular Generation},
  author={Halák, Matúš},
  year={2026},
  school={VU Amsterdam}
}