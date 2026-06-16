import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision.models import resnet18


# TRADES Loss Function

def trades_loss(model, x_natural, y, optimizer, step_size, epsilon, perturb_steps, beta, device):
    """
    Computes the TRADES loss: Cross-Entropy(clean) + beta * KL_Divergence(clean || adv)
    """
    #  KL-Divergence loss
    criterion_kl = nn.KLDivLoss(reduction='batchmean')
    

    model.eval()
    
    # Initializing adversarial examples with small  noise
    x_adv = x_natural.detach() + 0.001 * torch.randn(x_natural.shape).to(device).detach()
    x_adv = torch.clamp(x_adv, 0.0, 1.0) # Keep within valid image range
    
    # PGD  that the adversarial example that maximizes KL divergence loss
    for _ in range(perturb_steps):
        x_adv.requires_grad_()
        with torch.enable_grad():
            logits_clean = model(x_natural)
            logits_adv = model(x_adv)
            
            # maximizing  difference between clean and adversarial ouputs
            loss_kl = criterion_kl(F.log_softmax(logits_adv, dim=1),
                                   F.softmax(logits_clean, dim=1))
        
        grad = torch.autograd.grad(loss_kl, [x_adv])[0]
        
        x_adv = x_adv.detach() + step_size * torch.sign(grad.detach())
        x_adv = torch.min(torch.max(x_adv, x_natural - epsilon), x_natural + epsilon)
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
        
    model.train()
    
    optimizer.zero_grad()
    
    # final loss on the clean and generated adversarial 
    logits_clean = model(x_natural)
    logits_adv = model(x_adv)
    
    # Loss 1: Cross Entropy on clean data
    loss_natural = F.cross_entropy(logits_clean, y)
    
    # Loss 2: KL Divergence regularization
    loss_robust = criterion_kl(F.log_softmax(logits_adv, dim=1),
                               F.softmax(logits_clean, dim=1))
    
    # Combining them
    total_loss = loss_natural + beta * loss_robust
    return total_loss


def main():
    device = torch.device("cuda" if torch.cuda.is_available()  else "cpu")
    try:
        data = np.load("./train.npz")
        images = torch.from_numpy(data["images"]).float() / 255.0
        labels = torch.from_numpy(data["labels"]).long()
    except FileNotFoundError:
        print("ensure the file is in the current directory.")
        return
    
    dataset = TensorDataset(images, labels)
    val_size = int(0.1 * len(dataset))
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)


    NUM_CLASSES = 9

    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)
    model = model.to(device)

    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(1, 3, 32, 32).to(device)
        out = model(dummy_input)

  
    #  Hyperparameters for TRADES
    epochs = 70
    learning_rate = 0.01
    
    # Adversarial parameters for images scaled to [0, 1]
    epsilon = 8.0 / 255.0      # Maximum perturbation allowed
    step_size = 2.0 / 255.0    # Step size for PGD attack
    perturb_steps = 15         # Number of PGD steps
    beta = 5.0                 # Trade-off parameter (higher = more robust, less clean accuracy)

    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    # training loop
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            # Calculating TRADES loss
            loss = trades_loss(model, data, target, optimizer, 
                               step_size=step_size, epsilon=epsilon, 
                               perturb_steps=perturb_steps, beta=beta, device=device)
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            with torch.no_grad():
                outputs = model(data)
                _, predicted = outputs.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
                
        
        scheduler.step()
        
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
        print(f"Val Clean Acc: {val_acc:.2f}%\n")


    save_dir = "output"
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "model5.pt"))

if __name__ == "__main__":
    main()