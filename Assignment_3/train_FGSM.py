import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision.models import resnet18
import os

# 1. FGSM Attack Generation
def fgsm_attack(model, images, labels, epsilon, criterion, device):
    """
    Generates adversarial examples using the Fast Gradient Sign Method.
    """
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    
    # Require gradients for the input images
    images.requires_grad = True
    
    # Forward pass
    outputs = model(images)
    loss = criterion(outputs, labels)
    
    # Zero all existing gradients
    model.zero_grad()
    
    # Backward pass to calculate gradients of the image
    loss.backward()
    
    # Create the perturbed image by adjusting each pixel by epsilon in the direction of the gradient
    data_grad = images.grad.data
    perturbed_images = images + epsilon * data_grad.sign()
    
    # Clip to maintain valid pixel range [0, 1]
    perturbed_images = torch.clamp(perturbed_images, 0.0, 1.0)
    
    return perturbed_images.detach()

# 2. Main Training Loop
def train_fgsm():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load and split dataset
    print("Loading dataset...")
    data = np.load("./train.npz")
    images = torch.from_numpy(data["images"]).float() / 255.0
    labels = torch.from_numpy(data["labels"]).long()

    dataset = TensorDataset(images, labels)
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False)

    # Initialize model
    print("Initializing ResNet18...")
    NUM_CLASSES = 9
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(device)

    # Hyperparameters
    epochs = 50
    epsilon = 8.0 / 255.0  # Standard L_inf perturbation bound
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    print("\nStarting FGSM Adversarial Training...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (data_clean, target) in enumerate(train_loader):
            data_clean, target = data_clean.to(device), target.to(device)
            
            # Generate FGSM adversarial examples (model in eval mode for generation)
            model.eval()
            data_adv = fgsm_attack(model, data_clean, target, epsilon, criterion, device)
            model.train()
            
            optimizer.zero_grad()
            
            # Forward pass on both clean and adversarial images
            outputs_clean = model(data_clean)
            outputs_adv = model(data_adv)
            
            # Mixed loss to balance clean accuracy and robustness (50/50 split)
            loss_clean = criterion(outputs_clean, target)
            loss_adv = criterion(outputs_adv, target)
            loss = 0.5 * loss_clean + 0.5 * loss_adv
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch: {epoch}/{epochs} | Batch: {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
                
        scheduler.step()
        
        # Validation Phase
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for val_data, val_target in val_loader:
                val_data, val_target = val_data.to(device), val_target.to(device)
                outputs = model(val_data)
                _, predicted = outputs.max(1)
                total += val_target.size(0)
                correct += predicted.eq(val_target).sum().item()
                
        val_acc = 100. * correct / total
        print(f"==== Epoch {epoch} Summary ====")
        print(f"Train Loss: {train_loss/len(train_loader):.4f} | Val Clean Acc: {val_acc:.2f}%\n")


    save_dir = "outputs"
    os.makedirs(save_dir, exist_ok=True)

    print("Training complete. Saving model state dict...")
    torch.save(model.state_dict(), os.path.join(save_dir, "model_fgsm.pt"))
    print("Model successfully saved to outputs/model_fgsm.pt")

if __name__ == "__main__":
    train_fgsm()