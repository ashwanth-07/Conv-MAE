import os
import time
import argparse
import datetime
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Tuple
import json
import csv

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F

import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder

# Import your ConvMAE implementation
from convmae_models import ConvMAEPretrainer

from collections import defaultdict

class SimpleLogger:
    """Simple file-based logger for training metrics"""
    
    def __init__(self, log_dir: str, log_name: str = "training_log"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Create log files
        self.train_log_file = self.log_dir / f"{log_name}_train.csv"
        self.val_log_file = self.log_dir / f"{log_name}_val.csv"
        self.general_log_file = self.log_dir / f"{log_name}_general.txt"
        
        # Initialize CSV files with headers
        with open(self.train_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'step', 'loss', 'lr', 'timestamp'])
        
        with open(self.val_log_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['epoch', 'loss', 'timestamp'])
        
        # Start general log
        with open(self.general_log_file, 'w') as f:
            f.write(f"Training started at {datetime.datetime.now()}\n")
            f.write("="*50 + "\n")
    
    def log_train(self, epoch: int, step: int, loss: float, lr: float):
        """Log training metrics"""
        timestamp = datetime.datetime.now().isoformat()
        with open(self.train_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, step, loss, lr, timestamp])
    
    def log_val(self, epoch: int, loss: float):
        """Log validation metrics"""
        timestamp = datetime.datetime.now().isoformat()
        with open(self.val_log_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch, loss, timestamp])
    
    def log_message(self, message: str):
        """Log general message"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] {message}\n"
        print(log_entry.strip())  # Print to console
        
        with open(self.general_log_file, 'a') as f:
            f.write(log_entry)
    
    def log_epoch_summary(self, epoch: int, train_loss: float, val_loss: float = None, lr: float = None):
        """Log epoch summary"""
        message = f"Epoch {epoch:3d} | Train Loss: {train_loss:.6f}"
        if val_loss is not None:
            message += f" | Val Loss: {val_loss:.6f}"
        if lr is not None:
            message += f" | LR: {lr:.2e}"
        self.log_message(message)
    
    def plot_training_curves(self):
        """Plot and save training curves"""
        try:
            # Read training data
            train_data = []
            with open(self.train_log_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    train_data.append({
                        'epoch': int(row['epoch']),
                        'step': int(row['step']),
                        'loss': float(row['loss']),
                        'lr': float(row['lr'])
                    })
            
            # Read validation data
            val_data = []
            with open(self.val_log_file, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    val_data.append({
                        'epoch': int(row['epoch']),
                        'loss': float(row['loss'])
                    })
            
            if not train_data:
                return
            
            # Create plots
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            
            # Training loss over steps
            steps = [d['step'] for d in train_data]
            train_losses = [d['loss'] for d in train_data]
            axes[0, 0].plot(steps, train_losses, alpha=0.7)
            axes[0, 0].set_title('Training Loss vs Steps')
            axes[0, 0].set_xlabel('Steps')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].grid(True)
            
            # Learning rate over steps
            lrs = [d['lr'] for d in train_data]
            axes[0, 1].plot(steps, lrs)
            axes[0, 1].set_title('Learning Rate vs Steps')
            axes[0, 1].set_xlabel('Steps')
            axes[0, 1].set_ylabel('Learning Rate')
            axes[0, 1].grid(True)
            
            # Epoch-wise losses
            if val_data:
                train_epochs = sorted(set(d['epoch'] for d in train_data))
                val_epochs = [d['epoch'] for d in val_data]
                val_losses = [d['loss'] for d in val_data]
                
                # Average training loss per epoch
                epoch_train_losses = []
                for epoch in train_epochs:
                    epoch_losses = [d['loss'] for d in train_data if d['epoch'] == epoch]
                    epoch_train_losses.append(np.mean(epoch_losses))
                
                axes[1, 0].plot(train_epochs, epoch_train_losses, 'b-', label='Train')
                axes[1, 0].plot(val_epochs, val_losses, 'r-', label='Validation')
                axes[1, 0].set_title('Loss vs Epochs')
                axes[1, 0].set_xlabel('Epoch')
                axes[1, 0].set_ylabel('Loss')
                axes[1, 0].legend()
                axes[1, 0].grid(True)
                
                # Log scale
                axes[1, 1].semilogy(train_epochs, epoch_train_losses, 'b-', label='Train')
                axes[1, 1].semilogy(val_epochs, val_losses, 'r-', label='Validation')
                axes[1, 1].set_title('Loss vs Epochs (Log Scale)')
                axes[1, 1].set_xlabel('Epoch')
                axes[1, 1].set_ylabel('Loss (log)')
                axes[1, 1].legend()
                axes[1, 1].grid(True)
            
            plt.tight_layout()
            plot_path = self.log_dir / 'training_curves.png'
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            plt.close()
            
            self.log_message(f"Training curves saved to {plot_path}")
            
        except Exception as e:
            self.log_message(f"Error plotting training curves: {e}")

class ImageNetDataset:
    """ImageNet dataset loader for ConvMAE pretraining"""
    
    def __init__(
        self,
        data_path: str,
        split: str = 'train',
        image_size: int = 224,
        patch_size: int = 16,
        mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
        std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
    ):
        self.data_path = data_path
        self.split = split
        self.image_size = image_size
        self.patch_size = patch_size
        
        # MAE-style transforms - minimal augmentation for pretraining
        if split == 'train':
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(
                    image_size, 
                    scale=(0.2, 1.0), 
                    interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(
                    int(image_size * 1.14), 
                    interpolation=transforms.InterpolationMode.BICUBIC
                ),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std)
            ])
        
        # Create dataset
        dataset_path = os.path.join(data_path, split)
        self.dataset = ImageFolder(dataset_path, transform=self.transform)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        image, _ = self.dataset[idx]  # We don't need labels for MAE pretraining
        return image

def create_data_loaders(
    data_path: str,
    batch_size: int,
    num_workers: int = 8,
    image_size: int = 224,
    distributed: bool = False,
    world_size: int = 1,
    rank: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """Create train and validation data loaders"""
    
    # Create datasets
    train_dataset = ImageNetDataset(data_path, 'train', image_size)
    val_dataset = ImageNetDataset(data_path, 'val', image_size)
    
    # Create samplers for distributed training
    train_sampler = None
    val_sampler = None
    
    if distributed:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=world_size, rank=rank, shuffle=True
        )
        val_sampler = DistributedSampler(
            val_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader

def denormalize_image(
    tensor: torch.Tensor,
    mean: Tuple[float, float, float] = (0.485, 0.456, 0.406),
    std: Tuple[float, float, float] = (0.229, 0.224, 0.225)
) -> torch.Tensor:
    """Denormalize image tensor for visualization"""
    tensor = tensor.clone()
    for t, m, s in zip(tensor, mean, std):
        t.mul_(s).add_(m)
    return torch.clamp(tensor, 0, 1)

def visualize_reconstruction(
    original: torch.Tensor,
    masked: torch.Tensor,
    reconstructed: torch.Tensor,
    mask: torch.Tensor,
    save_path: str,
    num_samples: int = 8,
    patch_size: int = 16
):
    """Visualize original, masked, and reconstructed images"""
    
    device = original.device
    batch_size = min(num_samples, original.shape[0])
    
    # Denormalize images
    original_vis = denormalize_image(original[:batch_size].cpu())
    reconstructed_vis = denormalize_image(reconstructed[:batch_size].cpu())
    
    # Create masked version by applying mask to original image
    mask_resized = F.interpolate(
        mask[:batch_size].float().cpu(), 
        size=original.shape[-2:], 
        mode='nearest'
    )
    masked_vis = original_vis * mask_resized + 0.5 * (1 - mask_resized)  # Gray for masked regions
    
    # Create figure
    fig, axes = plt.subplots(3, batch_size, figsize=(batch_size * 3, 9))
    if batch_size == 1:
        axes = axes.reshape(3, 1)
    
    for i in range(batch_size):
        # Original image
        axes[0, i].imshow(original_vis[i].permute(1, 2, 0))
        axes[0, i].set_title('Original')
        axes[0, i].axis('off')
        
        # Masked image
        axes[1, i].imshow(masked_vis[i].permute(1, 2, 0))
        axes[1, i].set_title('Masked')
        axes[1, i].axis('off')
        
        # Reconstructed image
        axes[2, i].imshow(reconstructed_vis[i].permute(1, 2, 0))
        axes[2, i].set_title('Reconstructed')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def cosine_scheduler(
    base_value: float,
    final_value: float,
    epochs: int,
    niter_per_ep: int,
    warmup_epochs: int = 0,
    start_warmup_value: float = 0
):
    """Cosine learning rate scheduler with warmup"""
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule

def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    lr_schedule: np.ndarray,
    args: argparse.Namespace,
    logger: SimpleLogger
):
    """Train for one epoch"""
    model.train()
    metric_logger = defaultdict(AverageMeter)
    
    epoch_start_time = time.time()
    
    for step, images in enumerate(data_loader):
        step_start_time = time.time()
        
        # Adjust learning rate
        it = len(data_loader) * epoch + step
        if it < len(lr_schedule):
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_schedule[it]
        
        images = images.to(device, non_blocking=True)
        
        # Forward pass
        pred, loss, mask = model(images)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update metrics
        metric_logger['loss'].update(loss.item())
        metric_logger['lr'].update(optimizer.param_groups[0]["lr"])
        
        # Calculate step time
        step_time = time.time() - step_start_time
        metric_logger['step_time'].update(step_time)
        
        # Log to file
        if step % args.log_freq == 0:
            logger.log_train(epoch, it, loss.item(), optimizer.param_groups[0]["lr"])
        
        # Print progress
        if step % args.print_freq == 0:
            eta_seconds = step_time * (len(data_loader) - step)
            eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
            
            print(f'Epoch: [{epoch:3d}][{step:4d}/{len(data_loader):4d}] '
                  f'Loss: {metric_logger["loss"].avg:.6f} '
                  f'LR: {metric_logger["lr"].avg:.2e} '
                  f'Time: {step_time:.3f}s '
                  f'ETA: {eta_string}')
        
        # Visualize reconstruction
        if step % args.vis_freq == 0 and step > 0:
            save_path = os.path.join(args.output_dir, 'visualizations', f'train_epoch_{epoch}_step_{step}.png')
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            with torch.no_grad():
                visualize_reconstruction(
                    images, mask, pred, mask,
                    save_path, num_samples=8, patch_size=args.patch_size
                )
            
            logger.log_message(f"Training visualization saved: {save_path}")
    
    epoch_time = time.time() - epoch_start_time
    logger.log_message(f"Epoch {epoch} training completed in {epoch_time:.1f}s")
    
    return {k: v.avg for k, v in metric_logger.items()}

def validate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    epoch: int,
    args: argparse.Namespace,
    logger: SimpleLogger
):
    """Validate model"""
    model.eval()
    metric_logger = defaultdict(AverageMeter)
    
    val_start_time = time.time()
    
    with torch.no_grad():
        for step, images in enumerate(data_loader):
            images = images.to(device, non_blocking=True)
            
            # Forward pass
            pred, loss, mask = model(images)
            
            # Update metrics
            metric_logger['loss'].update(loss.item())
            
            # Visualize first batch
            if step == 0:
                save_path = os.path.join(args.output_dir, 'visualizations', f'val_epoch_{epoch}.png')
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                
                visualize_reconstruction(
                    images, mask, pred, mask,
                    save_path, num_samples=8, patch_size=args.patch_size
                )
                
                logger.log_message(f"Validation visualization saved: {save_path}")
            
            if step >= args.val_steps:  # Limit validation steps for efficiency
                break
    
    val_time = time.time() - val_start_time
    
    # Log validation metrics
    logger.log_val(epoch, metric_logger['loss'].avg)
    logger.log_message(f"Validation completed in {val_time:.1f}s")
    
    return {k: v.avg for k, v in metric_logger.items()}

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    args: argparse.Namespace,
    is_best: bool = False
):
    """Save checkpoint"""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'args': vars(args)  # Convert to dict for JSON serialization
    }
    
    # Save regular checkpoint
    checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    
    # Save best checkpoint
    if is_best:
        best_path = os.path.join(args.output_dir, 'checkpoint_best.pth')
        torch.save(checkpoint, best_path)
        print(f"✓ New best model saved with loss: {loss:.6f}")
    
    # Keep only last few checkpoints to save space
    if epoch > args.keep_checkpoints:
        old_checkpoint = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch - args.keep_checkpoints}.pth')
        if os.path.exists(old_checkpoint):
            os.remove(old_checkpoint)

def main():
    parser = argparse.ArgumentParser('ConvMAE pre-training')
    
    # Model parameters
    parser.add_argument('--backbone', default='b2', choices=['b0', 'b1', 'b2', 'b3'],
                        help='EfficientViT backbone model')
    parser.add_argument('--mask_ratio', default=0.75, type=float,
                        help='Masking ratio (percentage of removed patches)')
    parser.add_argument('--decoder_dim', default=512, type=int,
                        help='Decoder embedding dimension')
    parser.add_argument('--decoder_depth', default=8, type=int,
                        help='Decoder depth')
    parser.add_argument('--patch_size', default=16, type=int,
                        help='Patch size for reconstruction')
    
    # Training parameters
    parser.add_argument('--batch_size', default=256, type=int,
                        help='Batch size per GPU')
    parser.add_argument('--epochs', default=16, type=int,
                        help='Number of pre-training epochs')
    parser.add_argument('--warmup_epochs', default=4, type=int,
                        help='Warmup epochs')
    
    # Optimizer parameters
    parser.add_argument('--lr', default=1.5e-4, type=float,
                        help='Learning rate (absolute lr)')
    parser.add_argument('--min_lr', default=0., type=float,
                        help='Lower lr bound for cyclic schedulers that hit 0')
    parser.add_argument('--weight_decay', default=0.05, type=float,
                        help='Weight decay')
    
    # Dataset parameters
    parser.add_argument('--data_path', default='/content/imagenet-s/', type=str,
                        help='Dataset path')
    parser.add_argument('--image_size', default=224, type=int,
                        help='Input image size')
    parser.add_argument('--num_workers', default=8, type=int,
                        help='Number of data loading workers')
    
    # Logging and saving
    parser.add_argument('--output_dir', default='./output_convmae',
                        help='Path where to save, empty for no saving')
    parser.add_argument('--log_freq', default=50, type=int,
                        help='Log frequency')
    parser.add_argument('--print_freq', default=10, type=int,
                        help='Print frequency')
    parser.add_argument('--vis_freq', default=50, type=int,
                        help='Visualization frequency')
    parser.add_argument('--save_freq', default=50, type=int,
                        help='Save checkpoint frequency')
    parser.add_argument('--keep_checkpoints', default=5, type=int,
                        help='Number of checkpoints to keep')
    parser.add_argument('--val_steps', default=100, type=int,
                        help='Number of validation steps')
    
    # Resume training
    parser.add_argument('--resume', default='', type=str,
                        help='Resume from checkpoint')
    
    # Device
    parser.add_argument('--device', default='cuda',
                        help='Device to use for training')
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Save args
    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    # Initialize logger
    logger = SimpleLogger(args.output_dir, "convmae_training")
    
    logger.log_message(f"Job directory: {os.path.dirname(os.path.realpath(__file__))}")
    logger.log_message(f"Output directory: {args.output_dir}")
    logger.log_message(f"Arguments: {vars(args)}")
    
    device = torch.device(args.device)
    logger.log_message(f"Using device: {device}")
    
    # Create data loaders
    logger.log_message("Creating data loaders...")
    train_loader, val_loader = create_data_loaders(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
        distributed=False,  # Set to True for multi-GPU training
        world_size=1,
        rank=0
    )
    
    logger.log_message(f"Train dataset size: {len(train_loader.dataset)}")
    logger.log_message(f"Val dataset size: {len(val_loader.dataset)}")
    logger.log_message(f"Steps per epoch: {len(train_loader)}")
    
    # Create model
    logger.log_message("Creating model...")
    model = ConvMAEPretrainer(
        backbone_name=args.backbone,
        mask_ratio=args.mask_ratio,
        decoder_dim=args.decoder_dim,
        decoder_depth=args.decoder_depth,
        patch_size=args.patch_size
    )
    model.to(device)
    
    # Count parameters
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.log_message(f"Number of parameters: {n_parameters:,} ({n_parameters / 1e6:.1f}M)")
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Create learning rate schedule
    lr_schedule = cosine_scheduler(
        args.lr,
        args.min_lr,
        args.epochs,
        len(train_loader),
        warmup_epochs=args.warmup_epochs,
    )
    
    logger.log_message(f"Learning rate schedule: {args.lr} -> {args.min_lr} with {args.warmup_epochs} warmup epochs")
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_loss = float('inf')
    
    if args.resume:
        if os.path.isfile(args.resume):
            logger.log_message(f"Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint['epoch'] + 1  # Start from next epoch
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            best_loss = checkpoint.get('loss', float('inf'))
            logger.log_message(f"Resumed from epoch {start_epoch} with loss {best_loss:.6f}")
        else:
            logger.log_message(f"No checkpoint found at '{args.resume}'")
    
    logger.log_message("Starting training...")
    logger.log_message("="*80)
    
    training_start_time = time.time()
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start_time = time.time()
        
        # Train
        train_stats = train_one_epoch(
            model, train_loader, optimizer, device, epoch, lr_schedule, args, logger
        )
        
        # Validate
        val_stats = None
        if epoch % 10 == 0 or epoch == args.epochs - 1:  # Validate every 10 epochs
            val_stats = validate(model, val_loader, device, epoch, args, logger)
            
            # Save checkpoint
            is_best = val_stats['loss'] < best_loss
            if is_best:
                best_loss = val_stats['loss']
            
            if epoch % args.save_freq == 0 or epoch == args.epochs - 1 or is_best:
                save_checkpoint(model, optimizer, epoch, val_stats['loss'], args, is_best)
        
        # Log epoch summary
        epoch_time = time.time() - epoch_start_time
        if val_stats:
            logger.log_epoch_summary(epoch, train_stats['loss'], val_stats['loss'], train_stats['lr'])
        else:
            logger.log_epoch_summary(epoch, train_stats['loss'], lr=train_stats['lr'])
        
        logger.log_message(f"Epoch {epoch} completed in {epoch_time:.1f}s")
        
        # Plot training curves every 50 epochs
        if epoch % 50 == 0 and epoch > 0:
            logger.plot_training_curves()
    
    total_time = time.time() - training_start_time
    logger.log_message("="*80)
    logger.log_message(f"Training completed! Total time: {total_time:.1f}s ({total_time/3600:.1f}h)")
    logger.log_message(f"Best validation loss: {best_loss:.6f}")
    
    # Final plots
    logger.plot_training_curves()
    
    # Save final model
    final_checkpoint = {
        'epoch': args.epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': best_loss,
        'args': vars(args),
        'training_time': total_time
    }
    
    final_path = os.path.join(args.output_dir, 'checkpoint_final.pth')
    torch.save(final_checkpoint, final_path)
    logger.log_message(f"Final checkpoint saved: {final_path}")

if __name__ == '__main__':
    main()