import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import resnet34

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
        else:
            img = img.float() / 255.0
            
        return img, label

# Implementing Projected Gradient Descent L_inf Attack
def pgd_linf_attack(model, images, labels, epsilon, alpha, iters, criterion, device):
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    
    # Generating Adversarial samples within epsilon radius ball
    adv_images = images + torch.empty_like(images).uniform_(-epsilon, epsilon)
    adv_images = torch.clamp(adv_images, min=0.0, max=1.0).detach()
    
    # Optimizing adv_images to maximize the attack
    for _ in range(iters):
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


# Implementing Projected Gradient Descent L_2 Attack
def pgd_l2_attack(model, images, labels, epsilon, alpha, iters, criterion, device):
    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    
    # Generating adversarial samples
    adv_images = images.clone().detach() + (torch.randn_like(images) * 0.001)
    adv_images = torch.clamp(adv_images, 0.0, 1.0).detach()
    
    # Optimizing the adv_images to maximize the attack
    for _ in range(iters):
        adv_images.requires_grad = True
        outputs = model(adv_images)
        loss = criterion(outputs, labels)
        
        model.zero_grad()
        loss.backward()
        
        # Calculating the L2 norm of the gradient
        grad = adv_images.grad
        grad_norms = torch.norm(grad.view(grad.shape[0], -1), p=2, dim=1) + 1e-6
        grad = grad / grad_norms.view(-1, 1, 1, 1)
        
        # Modifying the adv_images
        adv_images = adv_images.detach() + alpha * grad
        delta = adv_images - images
        delta_norms = torch.norm(delta.view(delta.shape[0], -1), p=2, dim=1)
        factor = torch.min(torch.tensor(1.0).to(device), epsilon / (delta_norms + 1e-6))
        delta = delta * factor.view(-1, 1, 1, 1)

        adv_images = torch.clamp(images + delta, 0.0, 1.0).detach()
        
    return adv_images


# Implementing Projected Gradient Descent L_1 Attack
def pgd_l1_attack(model, images, labels, epsilon, alpha, iters, criterion, device):

    images = images.clone().detach().to(device)
    labels = labels.clone().detach().to(device)
    
    # Generating adversarial samples
    adv_images = images.clone().detach() + (torch.randn_like(images) * 0.001)
    adv_images = torch.clamp(adv_images, 0.0, 1.0).detach()

    # Optimizing the adv_images to maximize the attack
    for _ in range(iters):
        adv_images.requires_grad = True
        outputs = model(adv_images)
        loss = criterion(outputs, labels)
        
        model.zero_grad()
        loss.backward()

        # Calculating the L2 norm of the gradient
        grad = adv_images.grad
        grad_norms = torch.norm(grad.view(grad.shape[0], -1), p=1, dim=1) + 1e-6
        grad = grad / grad_norms.view(-1, 1, 1, 1)
        
        adv_images = adv_images.detach() + alpha * grad

        delta = adv_images - images
        delta_norms = torch.norm(delta.view(delta.shape[0], -1), p=1, dim=1)
        factor = torch.min(torch.tensor(1.0).to(device), epsilon / (delta_norms + 1e-6))
        delta = delta * factor.view(-1, 1, 1, 1)

        adv_images = torch.clamp(images + delta, 0.0, 1.0).detach()
        
    return adv_images


def train_robust_model():
    device = "cuda"
    # Using Automatic Mixed Precision to maximize the memory efficiency. 
    use_amp = device.type == 'cuda'
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    # Loading the dataset
    data = np.load("train.npz")
    images = torch.from_numpy(data["images"]) 
    labels = torch.from_numpy(data["labels"]).long()

    # Defining Augmentations specifically to fluctuate the pixel values
    transform_train = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomCrop(32, padding=4), 
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(), 
    ])

    val_size = int(0.1 * len(images))
    train_size = len(images) - val_size
    
    indices = torch.randperm(len(images)).tolist()
    train_idx, val_idx = indices[:train_size], indices[train_size:]

    train_dataset = AugmentedDataset(images[train_idx], labels[train_idx], transform=transform_train)
    val_dataset = AugmentedDataset(images[val_idx], labels[val_idx], transform=None)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=False, num_workers=2, pin_memory=True)

    NUM_CLASSES = 9 

    # Loading the model 
    model = resnet34(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(device)

    checkpoint_path = os.path.join("outputs", "best_model_pgd.pt")
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))


    # Defining the hyperparameters for all attacks
    epochs = 40 
    epsilon_linf = 8.0 / 255.0  
    alpha_linf = 2.0 / 255.0    
    
    epsilon_l2 = 0.5 
    alpha_l2 = 0.1   

    epsilon_l1 = 12.0
    alpha_l1 = 2.5
    
    iters = 7 
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    save_dir = "outputs"
    os.makedirs(save_dir, exist_ok=True)
    best_unified_score = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        
        for batch_idx, (data_clean, target) in enumerate(train_loader):
            data_clean, target = data_clean.to(device), target.to(device)
            
            model.eval()
            
            # For each batch, we can randomly select different attack
            attack_choice = random.choice(['linf', 'l2', 'l1'])
            if attack_choice == 'linf':
                data_adv = pgd_linf_attack(model, data_clean, target, epsilon_linf, alpha_linf, iters, criterion, device)
            elif attack_choice == 'l2':
                data_adv = pgd_l2_attack(model, data_clean, target, epsilon_l2, alpha_l2, iters, criterion, device)
            else:
                data_adv = pgd_l1_attack(model, data_clean, target, epsilon_l1, alpha_l1, iters, criterion, device)
                
            model.train()
            optimizer.zero_grad()
            
            with torch.autocast(device_type=device.type, enabled=use_amp):
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
                print(f"Epoch: {epoch}/{epochs} | Batch: {batch_idx}/{len(train_loader)} | Attack: {attack_choice} | Loss: {loss.item():.4f}")
                
        scheduler.step()
        

        model.eval()
        clean_correct, adv_linf_correct, total = 0, 0, 0
        
        for val_data, val_target in val_loader:
            val_data, val_target = val_data.to(device), val_target.to(device)
            
            with torch.no_grad():
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    outputs_clean = model(val_data)
                _, pred_clean = outputs_clean.max(1)
                clean_correct += pred_clean.eq(val_target).sum().item()
            
            with torch.enable_grad(): 
                val_data_adv = pgd_linf_attack(model, val_data, val_target, epsilon_linf, alpha_linf, iters, criterion, device)
                
            with torch.no_grad():
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    outputs_adv = model(val_data_adv)
                _, pred_adv = outputs_adv.max(1)
                adv_linf_correct += pred_adv.eq(val_target).sum().item()
                
            total += val_target.size(0)
                
        val_clean_acc = 100. * clean_correct / total
        val_adv_acc = 100. * adv_linf_correct / total
        
        # Unified Score: 0.5 * clean accuracy + 0.5 * robustness score
        unified_score = (0.5 * val_clean_acc) + (0.5 * val_adv_acc)
        
        print(f"============== Epoch {epoch} Summary ==============================")
        print(f"Train Loss: {train_loss/len(train_loader):.4f} | Clean Acc: {val_clean_acc:.2f}% | Robust Acc (L_inf): {val_adv_acc:.2f}% | Unified Score: {unified_score:.2f}")
        
        if unified_score > best_unified_score:
            best_unified_score = unified_score
            save_path = os.path.join(save_dir, "best_model_pgd.pt")
            torch.save(model.state_dict(), save_path)
            print(f"-> New best model saved with Score {best_unified_score:.2f}!\n")
        else:
            print("\n")

    torch.save(model.state_dict(), os.path.join(save_dir, "final_model_pgd.pt"))

if __name__ == "__main__":
    train_robust_model()