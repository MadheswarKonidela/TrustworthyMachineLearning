# TML_MIA
# Membership Inference Attack (MIA) 

**Team:** Madheswar Konidela & Pavan Kumar Matcha[cite: 1]  

## Overview
This repository contains the implementation of a Membership Inference Attack (MIA). The objective of an MIA is to determine whether a specific data sample was used in the training dataset of a given target model. 

For this project, we analyze a pre-trained `ResNet18` target model using a dataset containing both members and non-members drawn from the same distribution. The dataset is split into a public set (with membership labels) and a private set (without membership labels). 

### Evaluation Metric
The primary evaluation metric for this assignment is the True Positive Rate (TPR) at a strict 5% False Positive Rate (FPR) limit 
($Score = TPR @ 5\% FPR$). The goal is to maximize correct classifications while keeping misclassifications at or below 5%.

## Repository Contents
*   `rmia.py`: The main Python script implementing the Robust Membership Inference Attack (RMIA).
*   `*.csv` (Output File): The generated submission file containing the likelihood scores for the private dataset.

## Configuration & File Paths
Before running the code, you must update the file paths within `rmia.py` to point to your local or cloud environment directories. 

Open `rmia.py` and locate the `# --- Instructor Config ---` section at the top of the file. Update the following paths accordingly:

```python
# --- Instructor Config ---
from pathlib import Path

# Update these paths to where your datasets and target model are stored
PUB_PATH = Path("/path/to/your/pub.pt")       # Public dataset
PRIV_PATH = Path("/path/to/your/priv.pt")     # Private dataset
MODEL_PATH = Path("/path/to/your/model.pt")   # Pre-trained target model

# Update this path to where you want the final CSV to be saved
OUTPUT_CSV = Path("/path/to/your/output/submission.csv")
```


## Methodologies Explored
During the development of this project, multiple approaches were tested to find distinguishing signals between members and non-members:
*   **Gradient-Based:** Attempted to use the magnitude/norm of gradients, assuming non-members would generate higher loss and larger gradients. 
*   **Activations-Based:** Extracted activation energy from deeper layers of the network.
*   **Perturbation-Based:** Compared the logit scores and losses of original images against perturbed versions, testing if the model was more robust to perturbations on memorized data.
*   **Likelihood Ratio Attack (LiRA):** Implemented both offline (training 16 shadow models on public data) and online versions to categorize points using likelihood.


## Chosen Implementation: Robust Membership Inference Attack (RMIA)

**How it works:**
1.  **Data Splitting:** The public dataset is split into a reference training set and a population set (non-members unseen by any model).
2.  **Reference Model:** A new reference model ($R$) is trained on the reference set.
3.  **Baseline Calculation:** The population data is passed through both the target model ($T$) and the reference model ($R$) to calculate a baseline Log-Likelihood Ratio ( $LLR_{z} = Loss_{R}(z) - Loss_{T}(z)$ ). Since neither model saw this data, they should perform similarly.
4.  **Evaluation:** Private data points ($x$) are passed through both models to calculate $LLR_{x} = Loss_{R}(x) - Loss_{T}(x)$. If the target model memorized the point, its loss will be significantly lower, causing the LLR to deviate from the baseline. 

## Results & Conclusion
*   **Local Evaluation Score:** Consistently between `0.19` and `0.21`.
*   **Leaderboard Score:** `0.058655`.
