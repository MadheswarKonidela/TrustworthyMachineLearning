import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision.models import resnet18


# 1. TRADES Loss Function

def trades_loss(model, x_natural, y, optimizer, step_size, epsilon, perturb_steps, beta, device):
    """
    Computes the TRADES loss: Cross-Entropy(clean) + beta * KL_Divergence(clean || adv)
    """
    # Define KL-Divergence loss
    criterion_kl = nn.KLDivLoss(reduction='batchmean')
    
    # Set model to evaluation mode for generating adversarial examples
    model.eval()
    
    # Initialize adversarial examples with small random noise
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()
    x_adv = torch.clamp(x_adv, 0.0, 1.0) # Keep within valid image range
    
    # PGD loop to find the adversarial example that maximizes KL divergence
    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            logits_clean = model(x_natural)
            logits_adv = model(x_adv)
            
            # We want to maximize the difference between clean and adv predictions
            loss_kl = criterion_kl(F.log_softmax(logits_adv, dim=1),
                                   F.softmax(logits_clean, dim=1))
        
        # Calculate gradients
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        
        # Update adversarial example (gradient ascent on KL loss)
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        
        # Project back to the Linf epsilon-ball around the original image
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0) # Ensure valid pixel range
        
    # Set model back to train mode for the actual weight update
    model.train()
    
    # Zero gradients
    optimizer.zero_grad()
    
    # Calculate final loss on the clean and generated adversarial examples
    logits_clean = model(x_natural)
    logits_adv = model(x_adv)
    
    # Loss 1: Standard Cross Entropy on clean data
    loss_natural = F.cross_entropy(logits_clean, y)
    
    # Loss 2: KL Divergence regularization
    loss_robust = criterion_kl(F.log_softmax(logits_adv, dim=1),
                               F.softmax(logits_clean, dim=1))
    
    # Combine losses governed by the beta tradeoff parameter
    total_loss = loss_natural + beta * loss_robust
    return total_loss


# 2. Setup and Data Loading
def main():
    # Detect hardware
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    print("Loading dataset...")
    try:
        data = np.load("./train.npz")
        images = torch.from_numpy(data["images"]).float() / 255.0
        labels = torch.from_numpy(data["labels"]).long()
    except FileNotFoundError:
        print("Error: train.npz not found. Please ensure the file is in the current directory.")
        return

    # Create a 90/10 train/validation split to monitor clean accuracy
    dataset = TensorDataset(images, labels)
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    print(f"Total Dataset size: {len(dataset)}")
    print(f"Training on {train_size} samples, Validating on {val_size} samples")

# 3. Model Initialization
    NUM_CLASSES = 9
    
    print("Initializing ResNet18...")
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(device)

    # Sanity check: verify output shape
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 32, 32).to(device)
        out = model(dummy_input)
    print(f"Sanity Check - Output shape: {out.shape} (Expected: 1, 9)")

  
    # 4. Hyperparameters for TRADES
    epochs = 70
    learning_rate = 0.01
    
    # Adversarial parameters for images scaled to [0, 1]
    epsilon = 8.0 / 255.0      # Maximum perturbation allowed
    step_size = 2.0 / 255.0    # Step size for PGD attack
    perturb_steps = 15         # Number of PGD steps
    beta = 5.0                 # Trade-off parameter (higher = more robust, less clean accuracy)

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # 5. Training Loop
    print("\nStarting TRADES Training...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Calculate TRADES loss (this includes generating the adversarial examples)
            loss = trades_loss(model, data, target, optimizer, 
                               step_size=step_size, epsilon=epsilon, 
                               perturb_steps=perturb_steps, beta=beta, device=device)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Track clean training accuracy just for logging
            with torch.no_grad():
                outputs = model(data)
                _, predicted = outputs.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
            if batch_idx % 100 == 0:
                print(f"Epoch: {epoch}/{epochs} | Batch: {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
        
        scheduler.step()
        
        # Validation Phase (Clean Accuracy Check) 
        # The assignment strictly requires clean accuracy > 50%
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for val_data, val_target in val_loader:
                val_data, val_target = val_data.to(device), val_target.to(device)
                val_outputs = model(val_data)
                _, val_pred = val_outputs.max(1)
                val_total += val_target.size(0)
                val_correct += val_pred.eq(val_target).sum().item()
        
        val_acc = 100. * val_correct / val_total
        train_acc = 100. * correct / total
        print(f"==== Epoch {epoch} Summary ====")
        print(f"Train Loss: {train_loss/len(train_loader):.4f} | Train Clean Acc: {train_acc:.2f}%")
        print(f"Val Clean Acc: {val_acc:.2f}% (Must be > 50% for submission)\n")

    # 6. Save Submission Payload
    print("Training complete. Saving model state dict...")
    save_dir = "output"
    os.makedirs(save_dir, exist_ok=True)
    
    # Strict requirement: save only the state_dict [cite: 51]
    torch.save(model.state_dict(), os.path.join(save_dir, "model5.pt"))
    print(f"Model successfully saved to {save_dir}")

if __name__ == "__main__":
    main()