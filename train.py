import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import wandb
import torchvision
from torchvision import transforms
from PIL import Image
from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR

# Import model components (ensure these files have been updated as described)
from models.autoencoder import MiniSlotAutoencoder
from data.data_custom import get_dataloader

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

def train_epoch(model, train_loader, val_loader, optimizer, criterion, device, 
                current_epoch, start_global_step, log_interval=400):
    model.train()
    running_loss = 0.0
    global_step = start_global_step
    
    for epoch_step, batch in enumerate(train_loader):
        global_step += 1
        batch = batch.to(device)
        optimizer.zero_grad()
        
        recon, slots = model(batch)
        loss = criterion(recon, batch)
            
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        running_loss += loss.item()
        
        # Log training metrics per iteration.
        wandb.log({
            "train_loss": loss.item(),
            "grad_norm": grad_norm,
            "lr": optimizer.param_groups[0]['lr'],
            "global_step": global_step,
            "epoch": current_epoch + 1
        })
        print(f"Epoch {current_epoch+1} Step {epoch_step+1}, Global Step {global_step}, Loss: {loss.item():.4f}")
        
        if global_step % log_interval == 0:
            val_loss = validate_epoch(model, val_loader, criterion, device)
            wandb.log({"val_loss": val_loss, "global_step": global_step})
            print(f"--> Validation Loss at Global Step {global_step}: {val_loss:.4f}")
            save_val_examples(model, val_loader, device, current_epoch, global_step)
    avg_loss = running_loss / len(train_loader)
    return avg_loss, global_step

def validate_epoch(model, dataloader, criterion, device, max_batches=50):
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            batch = batch.to(device)
            recon, _ = model(batch)
            loss = criterion(recon, batch)
            running_loss += loss.item()
    return running_loss / max_batches

def save_val_examples(model, dataloader, device, epoch, global_step, num_examples=3):
    """
    Saves a few validation examples to wandb and locally in 'data/validation_output/'.
    For each example, the original and reconstructed images are concatenated side-by-side.
    """
    model.eval()
    output_dir = os.path.join("data", "validation_output")
    os.makedirs(output_dir, exist_ok=True)
    
    with torch.no_grad():
        batch = next(iter(dataloader))[:num_examples].to(device)
        recon, _ = model(batch)
        
        grid = torch.cat([batch, recon], dim=0)
        grid = torchvision.utils.make_grid(grid.cpu(), nrow=num_examples)
        
        # Save locally.
        local_path = os.path.join(output_dir, f"epoch_{epoch+1}_step_{global_step}_recon.png")
        torchvision.utils.save_image(grid, local_path)
        
        # Log to wandb.
        wandb.log({"reconstructions": wandb.Image(grid)}, step=global_step)
    model.train()

def combined_loss(recon, target):
    # Use simple MSE loss.
    return nn.MSELoss()(recon, target)

def main():
    wandb.init(project="SlotAttention", config={
        "learning_rate": 3e-4,
        "batch_size": 16,
        "num_epochs": 300,
        "num_slots": 10,
        "slot_dim": 128,
        "resolution": 128,
        "warmup_iters": 1000,
        "grad_clip": 1.0,
        "log_interval": 100
    })
    config = wandb.config
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Instantiate the autoencoder model.
    # Ensure that in your MiniSlotAutoencoder:
    # - SpatialBroadcastDecoder is initialized with resolution=(32,32) so that after two upsampling layers, the output is 128x128.
    # - SlotAttention is instantiated with iters=7.
    model = MiniSlotAutoencoder(
        num_slots=config.num_slots,
        slot_dim=config.slot_dim
    ).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=0.01)
    warmup = LinearLR(optimizer, start_factor=0.01, total_iters=config.warmup_iters)
    cosine = CosineAnnealingLR(optimizer, T_max=config.num_epochs)
    scheduler = SequentialLR(optimizer, [warmup, cosine], milestones=[config.warmup_iters])
    
    # Create data loaders using the custom data functions.
    train_loader = get_dataloader(root_dir="data/CLEVR_v1.0", split="train", batch_size=config.batch_size, shuffle=True, num_workers=4)
    val_loader = get_dataloader(root_dir="data/CLEVR_v1.0", split="val", batch_size=config.batch_size, shuffle=False, num_workers=4)

    start_epoch, global_step = 0, 0

    # Checkpoint Loading
    checkpoint_files = sorted(os.listdir(CHECKPOINT_DIR))
    if checkpoint_files:
        latest_ckpt = os.path.join(CHECKPOINT_DIR, checkpoint_files[-1])
        try:
            checkpoint = torch.load(latest_ckpt, map_location=device)
            model.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            start_epoch = checkpoint['epoch'] + 1
            global_step = checkpoint['global_step']
            print(f"Resuming from epoch {start_epoch}, step {global_step}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
    else:
        print("No checkpoint found. Starting fresh training.")

    for epoch in range(start_epoch, config.num_epochs):
        print(f"Epoch {epoch+1}/{config.num_epochs}")
        train_loss, global_step = train_epoch(model, train_loader, val_loader, optimizer, combined_loss, device, epoch, global_step, log_interval=config.log_interval)
        print(f"Epoch {epoch+1} complete. Average Train Loss: {train_loss:.4f}")
        wandb.log({"epoch_train_loss": train_loss, "epoch": epoch+1})
        
        scheduler.step()
        
        checkpoint = {
            'epoch': epoch,
            'global_step': global_step,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict()
        }
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"epoch_{epoch+1}.pth")
        torch.save(checkpoint, ckpt_path)
        wandb.save(ckpt_path)
        print(f"Checkpoint saved at epoch {epoch+1}.")

    print("Training complete.")

if __name__ == "__main__":
    main()
