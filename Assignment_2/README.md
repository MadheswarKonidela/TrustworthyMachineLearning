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


## Implementation:
Combination of Final Layer weight similarity, Confidence gap, Common Errors and Common bias between target model and suspect model. 
**Metric Used: final_score = max(weight_similarity, bias_agreement, error_agreement * 0.85 + confidence_gap * 0.15)**


## Results
*   **Leaderboard Score:** `0.629630`.
