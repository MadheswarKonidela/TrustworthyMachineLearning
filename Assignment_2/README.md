# TML_Assignment 2
# Model Stealing: Stolen Model Detection 

**Team:** Madheswar Konidela & Pavan Kumar Matcha

## Overview
This repository contains the implementation of a Stolen Model Detection. The objective of Stolen Model Detection is to determine whether a specific suspect model stole the functionality of our target model.

For this project, we profiled our pre-trained `ResNet18` target model in terms of its confidence gaps, it's mistakes and hidden biases. Similarity, we generate profiles of each suspect model and compared them.

### Evaluation Metric
The primary evaluation metric for this assignment is the True Positive Rate (TPR) at a 5% False Positive Rate (FPR) limit 
($Score = TPR @ 5\% FPR$). The goal is to maximize correct classifications while keeping misclassifications at or below 5%.

## Repository Contents
*   `Assignment_1.py`: The main Python script implementing the Model Ownership Resolution approaches.
*   `*.csv` (Output File): The generated submission file containing the likelihood scores for each suspect model to be stolen.

## Configuration & File Paths
Before running the code, you must update the file paths within `Assignment_2.py` to point to your local or cloud environment directories. 

Open `Assignment_2.py` and Update the following paths accordingly:

```python
from pathlib import Path

# Update these paths to where your dataset, train indices, target model, suspect models folder
data_root = "path/to/cifar100"
with open("path/to/train_main_idx.json", "r") as f:
    train_idx = json.load(f)
checkpoint_path = "path/to/target_model/weights.safetensors"
suspect_weights_path = list(Path("path/to/suspect_models").glob('*.safetensors'))

# Update this path to where you want the final CSV to be saved
submission_df.to_csv("path/to/your/output/submission.csv", index=False) 
```
**Execute the Script:**
   To install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   To run the code (after configuring all paths) 
   ```bash
   python Assignment_2.py
```


## Chosen Implementation: Robust Membership Inference Attack (RMIA)

**How it works:**
1.  **Data Splitting:** The public dataset is split into a reference training set and a population set (non-members unseen by any model).
2.  **Reference Model:** A new reference model ($R$) is trained on the reference set.
3.  **Baseline Calculation:** The population data is passed through both the target model ($T$) and the reference model ($R$) to calculate a baseline Log-Likelihood Ratio ( $LLR_{z} = Loss_{R}(z) - Loss_{T}(z)$ ). Since neither model saw this data, they should perform similarly.
4.  **Evaluation:** Private data points ($x$) are passed through both models to calculate $LLR_{x} = Loss_{R}(x) - Loss_{T}(x)$. If the target model memorized the point, its loss will be significantly lower, causing the LLR to deviate from the baseline. 

## Results & Conclusion
*   **Local Evaluation Score:** Consistently between `0.19` and `0.21`.
*   **Leaderboard Score:** `0.058655`.
