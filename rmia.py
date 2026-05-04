import os
import sys
import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.models import resnet18
import torchvision.transforms as transforms
from sklearn.metrics import roc_curve
from scipy.stats import norm

PUB_PATH = Path("/kaggle/input/datasets/madheswar/tml-mia/pub.pt")
PRIV_PATH = Path("/kaggle/input/datasets/madheswar/tml-mia/priv.pt")
MODEL_PATH = Path("/kaggle/input/datasets/madheswar/tml-mia/model.pt")
OUTPUT_CSV = Path("/kaggle/working/submission_rmia7.csv")

# Create the output directory if it doesn't exist
# OUTPUT_DIR = Path("output")
# OUTPUT_DIR.mkdir(exist_ok=True)
# OUTPUT_CSV = OUTPUT_DIR / "submission7.csv"

N_REF = 4 
EPOCHS = 30
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class TaskDataset(Dataset):
    def __init__(self, transform=None):
        self.ids = []
        self.imgs = []
        self.labels = []
        self.transform = transform
    def __getitem__(self, index):
        id_ = self.ids[index]
        img = self.imgs[index]
        if self.transform is not None:
            img = self.transform(img)
        label = self.labels[index]
        return id_, img, label
    def __len__(self):
        return len(self.ids)

class MembershipDataset(TaskDataset):
    def __init__(self, transform=None):
        super().__init__(transform)
        self.membership = []
    def __getitem__(self, index):
        id_, img, label = super().__getitem__(index)
        return id_, img, label, self.membership[index]

def get_model_arch():
    m = resnet18(weights=None)
    m.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
    m.maxpool = nn.Identity()
    m.fc = nn.Linear(512, 9)
    return m.to(DEVICE)

def logit_scaling(logits, labels):
    """
    Acts as log(P(y)) for our Likelihood calculations.
    """
    logits = logits / 2.0
    true_logits = torch.gather(logits, 1, labels.unsqueeze(1)).squeeze()
    mask = torch.ones_like(logits, dtype=torch.bool)
    mask.scatter_(1, labels.unsqueeze(1), False)
    wrong_logits = logits[mask].view(logits.size(0), -1)
    log_wrong_sum = torch.logsumexp(wrong_logits, dim=1)
    scaled_conf = true_logits - log_wrong_sum
    return scaled_conf

# --- Load Datasets ---
print("Loading datasets...")
pub_ds = torch.load(PUB_PATH, weights_only=False)
priv_ds = torch.load(PRIV_PATH, weights_only=False)

if isinstance(priv_ds, MembershipDataset):
    priv_ds.__class__ = TaskDataset 

MEAN, STD = [0.7406, 0.5331, 0.7059], [0.1491, 0.1864, 0.1301]
transform = transforms.Compose([
    transforms.Resize(32),
    transforms.Normalize(mean=MEAN, std=STD),
])
pub_ds.transform = transform
priv_ds.transform = transform


# RMIA ALGORITHM STEP 1: 
# UPDATED: We explicitly extract only the points where membership == 0.
# This guarantees that our Reference Model and our Z distribution are 
# 100% free of target model memorization.

print("Extracting ground-truth non-members for pure baseline...")
pub_labels = []
pub_loader_temp = DataLoader(pub_ds, batch_size=32, shuffle=False)
for _, _, _, membership in pub_loader_temp:
    pub_labels.extend(membership.numpy())
pub_labels = np.array(pub_labels)

# Isolate definitively unseen points
non_member_idx = np.where(pub_labels == 0)[0]

rng = np.random.RandomState(42)
perm = rng.permutation(len(non_member_idx))
shuffled_non_members = non_member_idx[perm]

split_idx = len(shuffled_non_members) - (len(shuffled_non_members) // 4)
ref_train_idx = shuffled_non_members[:split_idx] 
pop_idx = shuffled_non_members[split_idx:]       

print(f"Using {len(ref_train_idx)} unseen samples to train R")
print(f"Using {len(pop_idx)} unseen samples for Z population distribution")


# RMIA ALGORITHM STEP 2: TRAIN REFERENCE MODEL(S)

ref_scores_priv = np.zeros((N_REF, len(priv_ds)))
ref_scores_pop = np.zeros((N_REF, len(pop_idx)))
ref_scores_pub_all = np.zeros((N_REF, len(pub_ds))) 

for i in range(N_REF):
    print(f"Training Reference Model {i+1}/{N_REF}...")
    r_model = get_model_arch()
    
    loader = DataLoader(Subset(pub_ds, ref_train_idx), batch_size=32, shuffle=True)
    opt = torch.optim.Adam(r_model.parameters(), lr=1e-5)
    
    r_model.train()
    for epoch in range(EPOCHS):
        for _, imgs, labels, _ in loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            opt.zero_grad()
            loss = F.cross_entropy(r_model(imgs), labels)
            loss.backward()
            opt.step()
    
    r_model.eval()
    with torch.no_grad():
        # Get Pr(R(x)) for Private Data (The suspect points)
        priv_loader = DataLoader(priv_ds, batch_size=32, shuffle=False)
        pr_scores = [logit_scaling(r_model(imgs.to(DEVICE)), labels.to(DEVICE)).cpu() for _, imgs, labels in priv_loader]
        ref_scores_priv[i] = torch.cat(pr_scores).numpy()

        # Get Pr(R(z)) for Population Data (Z)
        pop_loader = DataLoader(Subset(pub_ds, pop_idx), batch_size=32, shuffle=False)
        po_scores = [logit_scaling(r_model(imgs.to(DEVICE)), labels.to(DEVICE)).cpu() for _, imgs, labels, _ in pop_loader]
        ref_scores_pop[i] = torch.cat(po_scores).numpy()

        # Get Pr(R(x)) for all Public Data (For local validation)
        pub_loader = DataLoader(pub_ds, batch_size=32, shuffle=False)
        pu_scores = [logit_scaling(r_model(imgs.to(DEVICE)), labels.to(DEVICE)).cpu() for _, imgs, labels, _ in pub_loader]
        ref_scores_pub_all[i] = torch.cat(pu_scores).numpy()

    del r_model
    torch.cuda.empty_cache()

# Average the reference scores 
avg_ref_priv = np.mean(ref_scores_priv, axis=0)
avg_ref_pop = np.mean(ref_scores_pop, axis=0)
avg_ref_pub_all = np.mean(ref_scores_pub_all, axis=0)



# RMIA ALGORITHM STEP 3: EVALUATE TARGET MODEL (T)

print("Evaluating Target Model...")
target_model = get_model_arch()
target_model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
target_model.eval()

target_scores_priv = []
target_scores_pop = []
target_scores_pub_all = []
priv_ids = []

with torch.no_grad():
    # 1. Target on Private
    for ids, imgs, labels in DataLoader(priv_ds, batch_size=32, shuffle=False):
        logits = target_model(imgs.to(DEVICE))
        target_scores_priv.append(logit_scaling(logits, labels.to(DEVICE)).cpu())
        priv_ids.extend(ids)
    target_scores_priv = torch.cat(target_scores_priv).numpy()

    # 2. Target on Population (Z)
    for _, imgs, labels, _ in DataLoader(Subset(pub_ds, pop_idx), batch_size=32, shuffle=False):
        logits = target_model(imgs.to(DEVICE))
        target_scores_pop.append(logit_scaling(logits, labels.to(DEVICE)).cpu())
    target_scores_pop = torch.cat(target_scores_pop).numpy()
    
    # 3. Target on All Public (for local validation)
    pub_true_membership = []
    for _, imgs, labels, membership in DataLoader(pub_ds, batch_size=32, shuffle=False):
        logits = target_model(imgs.to(DEVICE))
        target_scores_pub_all.append(logit_scaling(logits, labels.to(DEVICE)).cpu())
        pub_true_membership.extend(membership.numpy())
    target_scores_pub_all = torch.cat(target_scores_pub_all).numpy()


# RMIA ALGORITHM STEP 4: PAIRWISE LIKELIHOOD RATIO
print("Calculating RMIA Scores...")

# LLR_Z = LogPr(T(Z)) - LogPr(R(Z))
llr_population = target_scores_pop - avg_ref_pop

# Fit a distribution to the mathematically pure Population LLRs
mu_z = np.mean(llr_population)
std_z = max(np.std(llr_population), 1e-6)

# LLR_X = LogPr(T(X)) - LogPr(R(X))
llr_targets = target_scores_priv - avg_ref_priv

final_membership_scores = norm.cdf(llr_targets, loc=mu_z, scale=std_z)



# Prepare CSV
clean_ids = [str(i.item()) if torch.is_tensor(i) else str(i) for i in priv_ids]
df = pd.DataFrame({
    "id": clean_ids, 
    "score": final_membership_scores
})
df.to_csv(OUTPUT_CSV, index=False)
print(f"RMIA Submission saved to {OUTPUT_CSV}")



# LOCAL VALIDATION

print("Running Local Validation using RMIA...")

llr_local_val = target_scores_pub_all - avg_ref_pub_all

pub_rmia_scores = norm.cdf(llr_local_val, loc=mu_z, scale=std_z)

fpr, tpr, _ = roc_curve(pub_true_membership, pub_rmia_scores)
idx = np.argmin(np.abs(fpr - 0.05))
print(f"RMIA Local TPR @ 5% FPR: {tpr[idx]:.4f}")

import requests
import argparse
BASE_URL = "http://34.63.153.158"   #DONOT CHANGE
API_KEY = "7ac0b0cc7d5d1e6c8380088a58a60044"
TASK_ID = "01-mia"  #DONOT CHANGE
#submit
def die(msg):
    print(msg, file=sys.stderr)
    sys.exit(1)

parser = argparse.ArgumentParser(description="Submit a CSV file to the server.")
args, unknown = parser.parse_known_args()

submit_path = OUTPUT_CSV

if not submit_path.exists():
    die(f"File not found: {submit_path}")

try:
    with open(submit_path, "rb") as f:
        resp = requests.post(
            f"{BASE_URL}/submit/{TASK_ID}",
            headers={"X-API-Key": API_KEY},
            files={"file": (submit_path.name, f, "application/csv")},
            timeout=(10, 600),
        )
    try:
        body = resp.json()
    except Exception:
        body = {"raw_text": resp.text}

    if resp.status_code == 413:
        die("Upload rejected: file too large (HTTP 413).")

    resp.raise_for_status()

    print("Successfully submitted.")
    print("Server response:", body)
    submission_id = body.get("submission_id")
    if submission_id:
        print(f"Submission ID: {submission_id}")

except requests.exceptions.RequestException as e:
    detail = getattr(e, "response", None)
    print(f"Submission error: {e}")
    if detail is not None:
        try:
            print("Server response:", detail.json())
        except Exception:
            print("Server response (text):", detail.text)
    sys.exit(1)