import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, Dataset
from torchvision import transforms
from torchvision.models import resnet34


# 1. EMA (Exponential Moving Average) Class
class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}


# 2. TRADES Loss Function (with gradient clipping)
def trades_loss(model, x_natural, y, optimizer, step_size, epsilon, perturb_steps, beta, device, clip_grad=1.0):
    criterion_kl = nn.KLDivLoss(reduction='batchmean')
    model.eval()
    
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()
    x_adv = torch.clamp(x_adv, 0.0, 1.0)
    
    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            logits_clean = model(x_natural)
            logits_adv = model(x_adv)
            loss_kl = criterion_kl(F.log_softmax(logits_adv, dim=1),
                                   F.softmax(logits_clean, dim=1))
        
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        
        # Gradient clipping for stability
        if clip_grad is not None:
            grad = torch.clamp(grad, -clip_grad, clip_grad)
        
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        
    model.train()
    optimizer.zero_grad()
    
    logits_clean = model(x_natural)
    logits_adv = model(x_adv)
    
    loss_natural = F.cross_entropy(logits_clean, y)
    loss_robust = criterion_kl(F.log_softmax(logits_adv, dim=1),
                               F.softmax(logits_clean, dim=1))
    
    total_loss = loss_natural + beta * loss_robust
    return total_loss


# 3. Evaluation Utility (Robustness)
def eval_robust_accuracy(model, data_loader, epsilon, step_size, perturb_steps, device):
    model.eval()
    correct = 0
    total = 0
    
    for data, target in data_loader:
        data, target = data.to(device), target.to(device)
        
        x_adv = data.detach() + 0.001 * torch.randn(data.shape).to(device).detach()
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        
        for _ in range(perturb_steps):
            x_adv.requires_grad_()
            with torch.enable_grad():
                loss = F.cross_entropy(model(x_adv), target)
            grad = torch.autograd.grad(loss, [x_adv])[0]
            x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
            x_adv = torch.min(torch.max(x_adv, data - epsilon), data + epsilon)
            x_adv = torch.clamp(x_adv, 0.0, 1.0)
            
        with torch.no_grad():
            outputs = model(x_adv)
            _, predicted = outputs.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()
            
    return 100. * correct / total



# 4. Setup, Data Loading, and Augmentation
class AugmentedDataset(Dataset):
    def __init__(self, tensor_images, tensor_labels, transform=None):
        self.images = tensor_images
        self.labels = tensor_labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            img = self.transform(img)
        return img, label


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading dataset...")
    try:
        data = np.load("./train.npz")
        images = torch.from_numpy(data["images"]).float() / 255.0
        labels = torch.from_numpy(data["labels"]).long()
    except FileNotFoundError:
        print("Error: train.npz not found. Please ensure the file is in the current directory.")
        return

    val_size = int(0.1 * len(images))
    train_size = len(images) - val_size
    
    torch.manual_seed(42)
    np.random.seed(42)
    
    indices = torch.randperm(len(images))
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]

    images_train, labels_train = images[train_indices], labels[train_indices]
    images_val, labels_val = images[val_indices], labels[val_indices]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
    ])

    train_dataset = AugmentedDataset(images_train, labels_train, transform=train_transform)
    val_dataset = TensorDataset(images_val, labels_val)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

    print(f"Total Dataset size: {len(images)}")
    print(f"Training on {train_size} samples, Validating on {val_size} samples")


    # 5. Model Initialization (ResNet34)
    NUM_CLASSES = 9
    
    print("Initializing ResNet34 for increased capacity...")
    model = resnet34(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(device)

    ema = EMA(model, decay=0.999)

    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 32, 32).to(device)
        out = model(dummy_input)
    print(f"Sanity Check - Output shape: {out.shape} (Expected: 1, 9)")

    # 6. Hyperparameters (FIXED for TRADES stability)
    epochs = 60                 
    learning_rate = 0.001       
    
    epsilon = 8.0 / 255.0
    step_size = 2.0 / 255.0
    perturb_steps = 10            
    eval_perturb_steps = 20       
    beta = 4.0                    

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    save_dir = "output"
    os.makedirs(save_dir, exist_ok=True)
    best_weighted_score = 0.0


    # 7. Training Loop
    print("\nStarting TRADES + EMA Training...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            loss = trades_loss(model, data, target, optimizer, 
                               step_size=step_size, epsilon=epsilon, 
                               perturb_steps=perturb_steps, beta=beta, 
                               device=device, clip_grad=1.0)  # gradient clipping
            
            loss.backward()
            optimizer.step()
            
            ema.update(model)
            
            train_loss += loss.item()
            
            # Check for nan before continuing
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"WARNING: Loss is nan at epoch {epoch}, batch {batch_idx}")
                continue
            
            with torch.no_grad():
                outputs = model(data)
                _, predicted = outputs.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
            if batch_idx % 100 == 0:
                print(f"Epoch: {epoch}/{epochs} | Batch: {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")
        
        scheduler.step()
        

        # 8. Validation Phase (Using EMA Weights)
        ema.apply_shadow(model)
        model.eval()
        
        val_correct_clean = 0
        val_total = 0
        
        with torch.no_grad():
            for val_data, val_target in val_loader:
                val_data, val_target = val_data.to(device), val_target.to(device)
                val_outputs = model(val_data)
                _, val_pred = val_outputs.max(1)
                val_total += val_target.size(0)
                val_correct_clean += val_pred.eq(val_target).sum().item()
        
        val_clean_acc = 100. * val_correct_clean / val_total
        train_acc = 100. * correct / total
        
        val_robust_acc = 0.0
        if epoch % 5 == 0 or epoch > (epochs - 5):
            print("Evaluating Robust Accuracy against 20-step PGD...")
            val_robust_acc = eval_robust_accuracy(model, val_loader, epsilon, step_size, 
                                                  eval_perturb_steps, device)
        
        print(f"==== Epoch {epoch} Summary ====")
        print(f"Train Loss: {train_loss/len(train_loader):.4f} | Train Clean Acc (Active): {train_acc:.2f}%")
        print(f"Val Clean Acc (EMA):  {val_clean_acc:.2f}% (Must be > 50% for submission)")
        
        if epoch % 5 == 0 or epoch > (epochs - 5):
            print(f"Val Robust Acc (EMA): {val_robust_acc:.2f}%")
            
            current_weighted_score = (0.5 * val_clean_acc) + (0.5 * val_robust_acc)

            if current_weighted_score > best_weighted_score and val_clean_acc > 50.0:
                best_weighted_score = current_weighted_score
                save_path = os.path.join(save_dir, "model_trades_ema.pt")
                torch.save(model.state_dict(), save_path)
                print(f"--> New best weighted model saved! (Score: {current_weighted_score:.2f})")
                
        ema.restore(model)

    print("Training complete. The best EMA state dict is saved in the output directory.")


if __name__ == "__main__":
    main()