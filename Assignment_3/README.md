## TML_Assignment 3

## Adversarial ML: Robustness

**Team**: Pavan Kumar Matcha & Madheswar Konidela

## Overview
This repository contains the implementation of an adversarially robust image classifier. The objective of this assignment is to train a model that maintains high accuracy on both clean data and adversarially perturbed inputs, avoiding the pitfall of overfitting to a specific attack type.

For this project, we implemented a TRADES (TRadeoff-inspired Adversarial DEfense via Surrogate-loss minimization) training pipeline using a ResNet18 backbone and some other approaches but we got highest score in the leaderboard with TRADES.

## Evaluation Metric
Submissions are evaluated based on a unified score that equally weights standard performance and adversarial defense:
Score = 0.5 * Clean Accuracy + 0.5 * Robustness Score

**Constraint**: The model must achieve a clean accuracy strictly greater than 50% on the hidden test set to be accepted for evaluation.

## Repository Contents
`train_TRADES.py`: The main Python script implementing the TRADES.

`train_trades_augmented.p`₹: The main Python script implementing the TRADES training loop, EMA wrapper, and PGD evaluation.

`train_FGSM.py`: The main Python script implementing the FGSM.

`train_PGD_with_Aug.py`: The main Python script implementing the PGD on augmented data.

`PGD_with_Aug_+_L_inf_+_L_2_+_L_1.py`: he main Python script implementing the PGD on augmented data with combination of different L norms on different batches and combinig them.

`submission.py`: The script used to upload the model state dictionary to the evaluation server.

`model5.pt` (Output File): The generated state dictionary containing the weights of our robust model.

## Configuration & File Paths
Before running the code, you must update the file paths within `train_trades.py` to point to your local dataset and output directories.

Open `train_trades.py` and update the following paths accordingly:
```python
import numpy as np
import os

# Update this path to where your dataset is located
data = np.load("./path/to/train.npz")

# Update this path to where you want the final model weights to be saved
save_dir = "path/to/your/output"
save_path = os.path.join(save_dir, "model5.pt")
```

## Execute the Script
To install the dependencies:
```bash pip install torch torchvision numpy```

To run the training pipeline (after configuring all paths):
```bash python train_trades.py```

## Implementation
The training architecture relies on a ResNet18 model for the dual-objective learning process.
**Key Components**:
*   TRADES Regularization: The loss function is decomposed into two parts: standard Cross-Entropy on clean images (to maintain the >50% threshold) and a KL-Divergence penalty between clean and adversarial logits (to push the adversarial boundary).
*   Adversarial Generation: We utilized a 10-step Projected Gradient Descent (PGD) with random restarts within an $L_\infty$ bound of $\epsilon = 8/255$ to generate the inner-maximization examples during training.

## Results
**Leaderboard Score**: 0.513866