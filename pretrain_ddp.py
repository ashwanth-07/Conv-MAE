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
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Sampler
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F

from convmae_models import ConvMAEPretrainer, ConvMAEDecoder
from pretrain import (
    SimpleLogger, 
    ImageNetDataset, 
    AverageMeter, 
    cosine_scheduler, 
    visualize_reconstruction
)


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def init_distributed_mode(args):
    """Initialize distributed training environment."""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print(f'| distributed init (rank {args.rank}): {args.dist_url}', flush=True)
    dist.init_process_group(
        backend=args.dist_backend, 
        init_method=args.dist_url,
        world_size=args.world_size, 
        rank=args.rank
    )
    dist.barrier()
    setup_for_distributed(args.rank == 0)


def create_data_loaders_ddp(
    data_path: str,
    batch_size: int,
    num_workers: int = 8,
    image_size: int = 224
) -> Tuple[DataLoader, DataLoader, DistributedSampler, DistributedSampler]:
    train_dataset = ImageNetDataset(data_path, 'train', image_size)
    val_dataset = ImageNetDataset(data_path, 'val', image_size)
    
    train_sampler = DistributedSampler(train_dataset, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, shuffle=False)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    return train_loader, val_loader, train_sampler, val_sampler


def train_one_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    sampler: Sampler,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    lr_schedule: np.ndarray,
    args: argparse.Namespace,
    logger: SimpleLogger
):
    model.train()
    sampler.set_epoch(epoch) # Necessary for shuffling in DDP
    metric_logger = AverageMeter()
    
    epoch_start_time = time.time()
    
    for step, images in enumerate(data_loader):
        it = len(data_loader) * epoch + step
        if it < len(lr_schedule):
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_schedule[it]
        
        images = images.to(device, non_blocking=True)
        
        pred, loss, mask = model(images)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        dist.all_reduce(loss)
        metric_logger.update(loss.item() / get_world_size())
        
        if is_main_process() and step % args.log_freq == 0:
            logger.log_train(epoch, it, metric_logger.avg, optimizer.param_groups[0]["lr"])
        
        if is_main_process() and step % args.print_freq == 0:
            print(f'Epoch: [{epoch:3d}][{step:4d}/{len(data_loader):4d}] '
                  f'Loss: {metric_logger.avg:.6f} '
                  f'LR: {optimizer.param_groups[0]["lr"]:.2e}')

    epoch_time = time.time() - epoch_start_time
    if is_main_process():
        logger.log_message(f"Epoch {epoch} training completed in {epoch_time:.1f}s")
    
    return metric_logger.avg


def validate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    epoch: int,
    args: argparse.Namespace,
    logger: SimpleLogger
):
    model.eval()
    metric_logger = AverageMeter()
    
    with torch.no_grad():
        for step, images in enumerate(data_loader):
            images = images.to(device, non_blocking=True)
            pred, loss, mask = model(images)
            
            dist.all_reduce(loss)
            metric_logger.update(loss.item() / get_world_size())

            if is_main_process() and step == 0:
                save_path = os.path.join(args.output_dir, 'visualizations', f'val_epoch_{epoch}.png')
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                
                # model.module is required to access methods of the original model
                visualize_reconstruction(
                    images.cpu(), pred.cpu(), mask.cpu(), save_path,
                    args.norm_pix_loss, model.module.decoder,
                    num_samples=8, patch_size=args.patch_size
                )
                logger.log_message(f"Validation visualization saved: {save_path}")
            
            if step >= args.val_steps:
                break
    
    if is_main_process():
        logger.log_val(epoch, metric_logger.avg)
    
    return metric_logger.avg


def save_checkpoint(
    model: DDP,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    args: argparse.Namespace,
    is_best: bool = False
):
    if not is_main_process():
        return
        
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.module.state_dict(), # Save the underlying model
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'args': vars(args)
    }
    
    checkpoint_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch}.pth')
    torch.save(checkpoint, checkpoint_path)
    
    if is_best:
        best_path = os.path.join(args.output_dir, 'checkpoint_best.pth')
        torch.save(checkpoint, best_path)
        print(f"✓ New best model saved with loss: {loss:.6f}")


def main():
    parser = argparse.ArgumentParser('ConvMAE DDP pre-training')
    
    # Model parameters
    parser.add_argument('--backbone', default='b2', choices=['b0', 'b1', 'b2', 'b3'], help='EfficientViT backbone model')
    parser.add_argument('--mask_ratio', default=0.75, type=float, help='Masking ratio')
    parser.add_argument('--decoder_dim', default=512, type=int, help='Decoder embedding dimension')
    parser.add_argument('--decoder_depth', default=8, type=int, help='Decoder depth')
    parser.add_argument('--patch_size', default=16, type=int, help='Patch size')
    parser.add_argument('--norm_pix_loss', default=False, action='store_true', help='Use per-patch pixel normalization')
    
    # Training parameters
    parser.add_argument('--batch_size', default=90, type=int, help='Batch size per GPU')
    parser.add_argument('--epochs', default=200, type=int, help='Number of epochs')
    parser.add_argument('--warmup_epochs', default=10, type=int, help='Warmup epochs')
    parser.add_argument('--lr', default=1.5e-4, type=float, help='Learning rate')
    parser.add_argument('--min_lr', default=1e-6, type=float, help='Lower lr bound')
    parser.add_argument('--weight_decay', default=0.05, type=float, help='Weight decay')
    
    # Data and IO parameters
    parser.add_argument('--data_path', default='/path/to/imagenet', type=str, help='Dataset path')
    parser.add_argument('--image_size', default=224, type=int, help='Input image size')
    parser.add_argument('--num_workers', default=10, type=int, help='Number of data loading workers')
    parser.add_argument('--output_dir', default='./output_convmae_ddp', help='Path for output')
    parser.add_argument('--log_freq', default=50, type=int)
    parser.add_argument('--print_freq', default=10, type=int)
    parser.add_argument('--vis_freq', default=100, type=int, help='Visualization frequency')
    parser.add_argument('--save_freq', default=10, type=int)
    parser.add_argument('--val_steps', default=100, type=int)
    parser.add_argument('--resume', default='', type=str, help='Resume from checkpoint')
    
    # Distributed parameters
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')

    args = parser.parse_args()
    
    init_distributed_mode(args)
    
    device = torch.device(f'cuda:{args.gpu}')
    
    logger = SimpleLogger(args.output_dir, "convmae_training") if is_main_process() else None
    
    if is_main_process():
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        with open(os.path.join(args.output_dir, 'args.json'), 'w') as f:
            json.dump(vars(args), f, indent=2)
        logger.log_message(f"Arguments: {vars(args)}")
        logger.log_message(f"Using device: {device} | World size: {get_world_size()}")

    # Adjust LR and batch size for distributed training
    args.lr = args.lr * get_world_size() 
    
    train_loader, val_loader, train_sampler, _ = create_data_loaders_ddp(
        data_path=args.data_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_size=args.image_size,
    )
    if is_main_process():
        logger.log_message(f"Data loaders created. Train steps/epoch: {len(train_loader)}")

    model = ConvMAEPretrainer(
        backbone_name=args.backbone,
        mask_ratio=args.mask_ratio,
        decoder_dim=args.decoder_dim,
        decoder_depth=args.decoder_depth,
        patch_size=args.patch_size,
        norm_pix_loss=args.norm_pix_loss
    ).to(device)
    
    model = DDP(model, device_ids=[args.gpu], find_unused_parameters=True, broadcast_buffers=False )
    model_without_ddp = model.module

    if is_main_process():
        n_params = sum(p.numel() for p in model_without_ddp.parameters() if p.requires_grad)
        logger.log_message(f"Model created. Parameters: {n_params / 1e6:.1f}M")
    
    optimizer = torch.optim.AdamW(model_without_ddp.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    lr_schedule = cosine_scheduler(
        args.lr, args.min_lr, args.epochs, len(train_loader), warmup_epochs=args.warmup_epochs
    )
    
    start_epoch = 0
    best_loss = float('inf')
    
    if args.resume:
        if os.path.isfile(args.resume):
            if is_main_process():
                logger.log_message(f"Loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume, map_location=device)
            start_epoch = checkpoint['epoch'] + 1
            model_without_ddp.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            best_loss = checkpoint.get('loss', float('inf'))
            if is_main_process():
                logger.log_message(f"Resumed from epoch {start_epoch}")
        else:
             if is_main_process():
                logger.log_message(f"No checkpoint found at '{args.resume}'")

    if is_main_process():
        logger.log_message("Starting training...")

    for epoch in range(start_epoch, args.epochs):
        train_loss = train_one_epoch(
            model, train_loader, train_sampler, optimizer, device, epoch, lr_schedule, args, logger
        )
        
        val_loss = validate(model, val_loader, device, epoch, args, logger)

        if is_main_process():
            logger.log_epoch_summary(epoch, train_loss, val_loss, optimizer.param_groups[0]['lr'])
            
            is_best = val_loss < best_loss
            if is_best:
                best_loss = val_loss
            
            if epoch % args.save_freq == 0 or epoch == args.epochs - 1 or is_best:
                save_checkpoint(model, optimizer, epoch, val_loss, args, is_best)
            
            if epoch % 50 == 0 and epoch > 0:
                logger.plot_training_curves()

    if is_main_process():
        logger.log_message("Training complete.")
        logger.plot_training_curves()

if __name__ == '__main__':
    main()