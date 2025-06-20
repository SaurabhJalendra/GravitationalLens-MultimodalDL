# GravitationalLens-MultimodalDL

##  Project Overview

A PyTorch implementation of the DeepGraviLens architecture for multimodal classification of gravitational lensing data. This project fuses image data with light curve time series using CNNs and LSTMs to detect and classify different scales of gravitational lenses, enabling advances in astrophysical survey analysis and dark matter research.

Gravitational lensing, a phenomenon predicted by Einstein's theory of general relativity, occurs when the gravitational field of a massive object bends the light from a distant source. This project leverages deep learning to automatically classify these phenomena based on both visual imaging data and light curve time series measurements.

##  Scientific Background

Gravitational lenses are crucial astronomical phenomena that help scientists:
- Map dark matter distributions in the universe
- Discover distant galaxies that would otherwise be too faint to detect
- Constrain cosmological parameters and test theories of gravity
- Study the properties of lensing galaxies and galaxy clusters

The classification of gravitational lenses traditionally relies on visual inspection by experts, which becomes increasingly impractical with the growing volume of astronomical survey data. This project implements an automated approach that leverages both spatial (image) and temporal (light curve) information to improve classification accuracy.

##  Model Architecture

The DeepGraviLens architecture consists of three main components:

### 1. Image Branch (CNN)
- 4 convolutional blocks with progressive feature extraction:
  - Input: 4-channel images (45×45 pixels)
  - Block 1: 32 filters → (32×22×22)
  - Block 2: 64 filters → (64×11×11)
  - Block 3: 128 filters → (128×5×5)
  - Block 4: 256 filters → (256×1×1)
- Each block includes Conv2D, BatchNorm, ReLU, and MaxPooling

### 2. Time Series Branch (LSTM)
- 2-layer LSTM with 128 hidden units
- Input: Light curves (14 time steps × 4 features)
- Captures temporal patterns in brightness variations

### 3. Fusion and Classification
- Concatenation of features from both branches
- Two fully-connected layers (256 and 128 units) with ReLU and Dropout
- Final classification layer with 4 outputs (one per lensing class)

Total trainable parameters: 722,340

##  Dataset Description

The dataset consists of:
- **Image Data**: 4-channel images of size 45×45 pixels
- **Time Series Data**: Light curves with 14 time steps and 4 features per step
- **Classes**: 4 categories representing different gravitational lensing scenarios:
  1. Non-lens
  2. Galaxy-scale lens
  3. Group-scale lens
  4. Cluster-scale lens

The dataset is well-balanced with approximately 5,000 samples per class, split into training (~14,000 samples), validation (~3,000 samples), and testing (~3,000 samples) sets.

##  Implementation Details

### Dependencies
- PyTorch
- NumPy
- Matplotlib
- scikit-learn
- tqdm
- pandas

### Training Strategy
- Loss function: Cross-entropy loss
- Optimizer: Adam with learning rate 0.001
- Learning rate scheduler: ReduceLROnPlateau
- Regularization: Dropout (0.3-0.5) and BatchNormalization
- Early stopping to prevent overfitting
- Batch size: 64
- Training for up to 30 epochs with early stopping

##  Results

The model achieves significant improvements over traditional single-modality approaches:
- Higher accuracy in distinguishing between different scales of gravitational lenses
- Improved robustness to noise and observational artifacts
- Better generalization to unseen data

The multimodal approach demonstrates the value of combining spatial and temporal information for astronomical classification tasks.

##  Usage Instructions

### Setup
```bash
# Clone the repository
git clone https://github.com/SaurabhJalendra/GravitationalLens-MultimodalDL.git
cd GravitationalLens-MultimodalDL

# Install requirements
pip install torch numpy matplotlib scikit-learn tqdm pandas
```

### Running the Notebook
Open and run the Jupyter notebook:
```bash
jupyter notebook DL_assignment_2B_group-111.ipynb
```

The notebook contains all the code for data preparation, model definition, training, and evaluation.

##  Contributors
- Group 111 - Deep Learning Team

##  License
This project is provided for educational purposes only.

##  References
- Gravitational lensing theory: https://en.wikipedia.org/wiki/Gravitational_lens
- Original DeepGraviLens paper: "DeepGraviLens: a multi-modal architecture for classifying gravitational lensing data"
