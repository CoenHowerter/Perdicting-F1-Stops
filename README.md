# Formula 1 Pit Stop Prediction: Kaggle Playground Series 2026 (S6E5)

PyTorch-based deep learning solution for the May 2026 Kaggle Tabular Playground challenge. The goal is to predict whether a Formula 1 driver will pit on the next lap based on race conditions and historical features.

## Performance Summary
* Validation Strategy: 5-Fold Stratified Cross-Validation
* Mean Fold ROC AUC: 0.93873
* Overall Out-of-Fold (OOF) ROC AUC: 0.93859

## Model Architecture
The model is a Tabular Neural Network with embedded categorical entities:
* Categorical Embeddings: Driver (50-dim), Race (13-dim), and Compound (3-dim).
* Multi-Layer Perceptron (MLP): 
  * Input Size: 77 (66 embedded dims + 11 continuous features)
  * Layer 1: Linear (77 -> 128) -> BatchNorm1d -> ReLU -> Dropout(0.3)
  * Layer 2: Linear (128 -> 64) -> BatchNorm1d -> ReLU -> Dropout(0.3)
  * Layer 3: Linear (64 -> 1) -> Sigmoid Output

## Data Processing
* Missing Value Imputation: Median values for continuous features; 'Missing' category for categorical indicators.
* Feature Normalization: Continuous values standardized using StandardScaler.

## Training Details
* Batch Size: 512
* Epochs: 30
* Optimizer: Adam (Learning Rate: 1e-3, Weight Decay: 1e-5)
* Early Stopping Patience: 5 epochs

## Repository Structure
* .gitignore
* README.md
* F1_Pit_Stop_Prediction.ipynb
