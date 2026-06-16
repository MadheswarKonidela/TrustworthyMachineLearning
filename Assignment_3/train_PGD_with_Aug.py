import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision.models import resnet34
import os
import torch.optim as optim
from torchvision import transforms

# Defining a class to perform online Data Augmentation
class AugmentedDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label

# Implementing Projected Gradient Descent Attack
def pgd_attack(model, images, labels, epsilon, alpha, iters, criterion, device):
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    
    # Generating Adversarial samples within epsilon radius ball
    adv_images = images + torch.empty_like(images).uniform_(-epsilon, epsilon)
    adv_images = torch.clamp(adv_images, min=0.0, max=1.0).detach()
    
    # Optimizing adv_images to maximize the attack
    for i in range(iters):
        adv_images.requires_grad = True
        outputs = model(adv_images)
        loss = criterion(outputs, labels)
        
        model.zero_grad()
        loss.backward()
        
        # Moving in the direction to increase the loss
        adv_images = adv_images + alpha * adv_images.grad.sign()
        
        eta = torch.clamp(adv_images - images, min=-epsilon, max=epsilon)
        
        adv_images = torch.clamp(images + eta, min=0.0, max=1.0).detach()
            
    return adv_images


def train_pgd():
    device = "cuda"
    # Using Automatic Mixed Precision to maximze the memory efficiency.    
    use_amp = device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Loading the dataset
    data = np.load("train.npz")
    images = torch.from_numpy(data["images"]).float() / 255.0
    labels = torch.from_numpy(data["labels"]).long()

    # Performing some simple augmentations to artificially expand the dataset
    transform_train = transforms.Compose([
        transforms.RandomCrop(images.shape[-1], padding=4), 
        transforms.RandomHorizontalFlip(),
    ])

    val_size = int(0.1 * len(images))
    train_size = len(images) - val_size
    
    indices = torch.randperm(len(images)).tolist()
    train_idx, val_idx = indices[:train_size], indices[train_size:]

    # Generating the augmented samples
    train_dataset = AugmentedDataset(images[train_idx], labels[train_idx], transform=transform_train)
    val_dataset = AugmentedDataset(images[val_idx], labels[val_idx], transform=None) # No augmentation for val

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=2)
    NUM_CLASSES = 9
    
    # Loading the model
    model = resnet34(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(device)

    # Defining Hyperparameters
    epochs = 20
    epsilon = 8.0 / 255.0  # Maximum perturbation allowed
    alpha = 2.0 / 255.0    # Step size
    iters = 7              # Number of PGD iterations
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    save_dir = "outputs"
    os.makedirs(save_dir, exist_ok=True)
    best_val_adv_acc = 0.0


    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (data_clean, target) in enumerate(train_loader):
            data_clean, target = data_clean.to(device), target.to(device)
            
            model.eval()
            # Generating the Adversarial Data
            data_adv = pgd_attack(model, data_clean, target, epsilon, alpha, iters, criterion, device)
            model.train()
            
            optimizer.zero_grad()
            
            with torch.autocast(device_type=device.type, enabled=use_amp):
                # Getting both unchanged input data predictions and modified input data predictions
                outputs_clean = model(data_clean)
                outputs_adv = model(data_adv)
                
                loss_clean = criterion(outputs_clean, target)
                loss_adv = criterion(outputs_adv, target)
                
                # Calculating the combined loss
                loss = 0.5 * loss_clean + 0.5 * loss_adv
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss.item()
            
            if batch_idx % 100 == 0:
                print(f"Epoch: {epoch}/{epochs} | Batch: {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
                
        scheduler.step()

        model.eval()
        clean_correct, adv_correct, total = 0, 0, 0
        
        for val_data, val_target in val_loader:
            val_data, val_target = val_data.to(device), val_target.to(device)
            
            with torch.no_grad():
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    outputs_clean = model(val_data)
                _, pred_clean = outputs_clean.max(1)
                clean_correct += pred_clean.eq(val_target).sum().item()
            
            with torch.enable_grad(): 
                val_data_adv = pgd_attack(model, val_data, val_target, epsilon, alpha, iters, criterion, device)
                
            with torch.no_grad():
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    outputs_adv = model(val_data_adv)
                _, pred_adv = outputs_adv.max(1)
                adv_correct += pred_adv.eq(val_target).sum().item()
                
            total += val_target.size(0)
                
        val_clean_acc = 100. * clean_correct / total
        val_adv_acc = 100. * adv_correct / total
        
        print(f"==== Epoch {epoch} Summary =====================")
        print(f"Train Loss: {train_loss/len(train_loader):.4f} | Val Clean Acc: {val_clean_acc:.2f}% | Val Robust Acc: {val_adv_acc:.2f}%")
        
        if val_adv_acc > best_val_adv_acc:
            best_val_adv_acc = val_adv_acc
            save_path = os.path.join(save_dir, "best_model_pgd.pt")
            torch.save(model.state_dict(), save_path)
            print(f"-> New best robust model saved with {best_val_adv_acc:.2f}% robust accuracy!\n")
        else:
            print("\n")

    torch.save(model.state_dict(), os.path.join(save_dir, "final_model_pgd.pt"))

if __name__ == "__main__":
    train_pgd()