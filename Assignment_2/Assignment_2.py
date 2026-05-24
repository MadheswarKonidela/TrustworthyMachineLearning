import os
import sys
import requests
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.models import resnet18
from safetensors.torch import load_file
import pandas as pd

from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
import json
import numpy as np


device = "cuda" if torch.cuda.is_available() else "cpu"

# --------------------------------
# LOADING A MODEL (EXAMPLE: TARGET MODEL)
# --------------------------------
def make_model():
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, 100)
    return model

"""
Implementing the target model's unique biased random cropping fingerprint
according to the augmentations used while training.
"""
class BiasedCropTransform:
    def __init__(self, bias_x=0.5, bias_y=-0.25, jitter=0.25, ref_padding=4):
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.jitter = jitter
        self.ref_padding = ref_padding

    def __call__(self, img):
        pad_transform = transforms.Pad(self.ref_padding, padding_mode='reflect')
        padded_img = pad_transform(img)
        
        base_x = 4 + int(self.bias_x * 4)
        base_y = 4 + int(self.bias_y * 4)
        
        j_range = int(self.jitter * 4)
        if j_range > 0:
            dx = np.random.randint(-j_range, j_range + 1)
            dy = np.random.randint(-j_range, j_range + 1)
        else:
            dx, dy = 0, 0
            
        crop_x = max(0, min(base_x + dx, 8))
        crop_y = max(0, min(base_y + dy, 8))
        
        return transforms.functional.crop(padded_img, crop_y, crop_x, 32, 32)

"""
Function to get the weights of weights of fully connected layer at last
to measure the similarity of the inference mechanism.
"""
def get_fc_weights(state_dict):
    return state_dict['fc.weight'].view(-1).to(device)

"""
To Extract the confidences, predictions, and labels of the model.
"""
def extract_model_outputs(model, dataloader, seed=None):
    # To ensure the augmentations are identical.
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
        
    confidences = []
    predictions = []
    labels_list = []
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs = inputs.to(device)
            logits = model(inputs)
            probs = F.softmax(logits, dim=1)
            max_probs, preds = torch.max(probs, dim=1)
            
            confidences.append(max_probs)
            predictions.append(preds)
            labels_list.append(labels)
            
    return torch.cat(confidences), torch.cat(predictions), torch.cat(labels_list).to(device)

"""
Loading the dataset and normalizing as specified.
"""
data_root = r"data\cifar-100-python"

norm_mean = (0.5071, 0.4867, 0.4408)
norm_std = (0.2675, 0.2565, 0.2761)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])


"""
Initializing the biased transform.
"""
transform_biased = transforms.Compose([
    BiasedCropTransform(bias_x=0.5, bias_y=-0.25, jitter=0.25),
    transforms.ToTensor(),
    transforms.Normalize(norm_mean, norm_std),
])

# To perform Membership Inference Attack.
"""
Loading the Target model's Training Subset (Members)
"""

full_train_dataset = datasets.CIFAR100(root=data_root, train=True, download=True, transform=transform)
with open(r"target_model\train_main_idx.json", "r") as f:
    train_idx = json.load(f)

train_subset = Subset(full_train_dataset, train_idx) 
train_loader = DataLoader(dataset=train_subset, batch_size=64, shuffle=False)

"""
Loading the rest of the Training Dataset (Non-Members).
"""
all_train_idx = set(range(len(full_train_dataset)))
non_member_idx = list(all_train_idx - set(train_idx))
non_member_subset = Subset(full_train_dataset, non_member_idx)
non_member_loader = DataLoader(dataset=non_member_subset, batch_size=64, shuffle=False)

# To check for biased behavior.
"""
Loading the Test Dataset and performing normal transform as well as biased transform.
"""
test_dataset_clean = datasets.CIFAR100(root=data_root, train=False, download=True, transform=transform)
test_loader_clean = DataLoader(dataset=test_dataset_clean, batch_size=64, shuffle=False)

test_dataset_biased = datasets.CIFAR100(root=data_root, train=False, download=True, transform=transform_biased)
test_loader_biased = DataLoader(dataset=test_dataset_biased, batch_size=64, shuffle=False)


# Loading the model weights.
checkpoint_path = r"target_model\weights.safetensors"
target_state_dict = load_file(checkpoint_path, device=device) 
target_model = make_model() 
target_model.load_state_dict(target_state_dict, strict=True)
target_model.eval()
target_model = target_model.to(device)
target_fc_weights = get_fc_weights(target_state_dict)


# Establishing the target model's baseline behavior.
target_train_conf, _, _ = extract_model_outputs(target_model, train_loader)
target_non_member_conf, _, _ = extract_model_outputs(target_model, non_member_loader)

# Baseline overfitting gap: High confidence on seen data vs lower confidence on unseen data
target_gap = target_train_conf.mean().item() - target_non_member_conf.mean().item()

# Comparing the biased behavior of the target model on test dataset.
target_test_conf, target_preds_clean, target_labels_clean = extract_model_outputs(target_model, test_loader_clean)
_, target_preds_biased, _ = extract_model_outputs(target_model, test_loader_biased, seed=42)

# Identifying where the target model makes mistakes for ModelDiff error agreement
target_errors_mask = (target_preds_clean != target_labels_clean)


# Evaluating the suspect models
suspect_weights_path = list(Path(r"suspect_models").glob('*.safetensors'))

# We will be evaluating using multiple metrics to detect suspect models trained using multiple model stealing attacks.
weight_similarities = []
gap_distances = []
error_agreements = []
biased_agreements = []

for count, each_path in enumerate(suspect_weights_path):
    suspect_state_dict = load_file(each_path, device=device)
    suspect_model = make_model()
    suspect_model.load_state_dict(suspect_state_dict, strict=True)
    suspect_model.eval()
    suspect_model = suspect_model.to(device)
    
    # Metric 1: FC Weight Cosine Similarity
    suspect_fc_weights = get_fc_weights(suspect_state_dict)
    cos_sim = F.cosine_similarity(target_fc_weights.unsqueeze(0), suspect_fc_weights.unsqueeze(0)).item()
    weight_similarities.append(cos_sim)

    # Metric 2: Specific Training Subset Membership Gap 
    suspect_train_conf, _, _ = extract_model_outputs(suspect_model, train_loader)
    suspect_non_member_conf, _, _ = extract_model_outputs(suspect_model, non_member_loader)
    suspect_gap = suspect_train_conf.mean().item() - suspect_non_member_conf.mean().item()
    
    # If the suspect overfits heavily to the target model, its confidence gap will be similar to the target model as well. 
    gap_distance = abs(target_gap - suspect_gap)
    gap_distances.append(gap_distance)

    # Metric 3: ModelDiff Error Agreement (looking for same mistaking in target as well as suspect models) 
    _, suspect_preds_clean, _ = extract_model_outputs(suspect_model, test_loader_clean)
    if target_errors_mask.sum() > 0:
        error_agree = (target_preds_clean[target_errors_mask] == suspect_preds_clean[target_errors_mask]).float().mean().item()
    else:
        error_agree = 0.0
    error_agreements.append(error_agree)

    # Metric 4: Biased Crop Data Fingerprinting
    _, suspect_preds_biased, _ = extract_model_outputs(suspect_model, test_loader_biased, seed=42)
    biased_agree = (target_preds_biased == suspect_preds_biased).float().mean().item()
    biased_agreements.append(biased_agree)    
    del suspect_model, suspect_state_dict, suspect_fc_weights

# Normalizing the results for fair comparisons and comparing all metrics to form a single metric.
weight_similarities = np.array(weight_similarities)
gap_distances = np.array(gap_distances)
error_agreements = np.array(error_agreements)
biased_agreements = np.array(biased_agreements)

# Normalizing the baseline gap footprint (invert so that minimum distance maps to a score of 1.0)
min_gap = gap_distances.min()
max_gap = gap_distances.max()
if max_gap != min_gap:
    gap_scores = 1.0 - ((gap_distances - min_gap) / (max_gap - min_gap))
else:
    gap_scores = np.ones_like(gap_distances)

# Constructing final stealing confidence scores
confidence_scores = []
for i in range(len(suspect_weights_path)):
    w_s = max(0.0, weight_similarities[i])
    e_a = error_agreements[i]
    b_a = biased_agreements[i]
    g_s = gap_scores[i]
    
    # comparing multiple metrics and if a single metric in very high then we will consider it.
    final_score = max(w_s, b_a, e_a * 0.85 + g_s * 0.15)
    
    # Clipping securely within continuous range bounds
    final_score = max(0.001, min(0.999, final_score))
    confidence_scores.append(final_score)


# # --------------------------------
# # SUBMISSION FORMAT
# # --------------------------------

"""
The submission must be a .csv file with the following format:

-"id": ID of the subset (from 0 to 359)
-"score": Stealing confidence score for each model (float)
"""
submission_df = pd.DataFrame({
    "id": list(range(len(confidence_scores))),
    "score": confidence_scores
})
submission_df.to_csv("submission.csv", index=False) 