# Modifications Copyright © 2025 Advanced Micro Devices, Inc. All rights reserved.
#

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

from efficientvit.models.utils import build_kwargs_from_config

__all__ = ["LayerNorm2d", "BatchNorm2d", "build_norm", "reset_bn", "set_norm_eps"]


class LayerNorm2d(nn.LayerNorm):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x - torch.mean(x, dim=1, keepdim=True)
        out = out / torch.sqrt(torch.square(out).mean(dim=1, keepdim=True) + self.eps)
        if self.elementwise_affine:
            out = out * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)
        return out

class BatchNorm2d(nn.Module):
    """
    BatchNorm2d that can optionally normalize only valid (unmasked) pixels.
    When no mask is provided, behaves identically to nn.BatchNorm2d.
    """
    
    def __init__(self, num_features, eps=1e-5, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        
        # Learnable parameters
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        
        # Running statistics
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))
        
    def forward(self, x, mask=None):
        """
        Args:
            x: Input tensor (B, C, H, W)
            mask: Optional boolean mask (B, H, W) where True = valid pixel
                  If None, applies standard batch normalization to all pixels
        """
        B, C, H, W = x.shape
        
        # If no mask provided, use standard batch normalization
        if mask is None:
            if self.training:
                # Standard batch norm computation
                mean = x.mean(dim=(0, 2, 3))  # (C,)
                var = x.var(dim=(0, 2, 3), unbiased=False)  # (C,)
                
                # Update running statistics
                self.running_mean.mul_(1 - self.momentum).add_(mean.detach(), alpha=self.momentum)
                self.running_var.mul_(1 - self.momentum).add_(var.detach(), alpha=self.momentum)
            else:
                # Use running statistics
                mean = self.running_mean
                var = self.running_var
            
            # Normalize all pixels
            mean = mean.view(1, C, 1, 1)
            var = var.view(1, C, 1, 1)
            weight = self.weight.view(1, C, 1, 1)
            bias = self.bias.view(1, C, 1, 1)
            
            return (x - mean) / torch.sqrt(var + self.eps) * weight + bias
        
        # Masked batch normalization - always compute from valid pixels when mask is provided
        # Expand mask to match input: (B, H, W) -> (B, C, H, W)
        if mask.dim() == 3:
            mask = mask.unsqueeze(1).expand(B, C, H, W)
        
        # Count valid pixels per channel
        valid_count = mask.sum(dim=(0, 2, 3))  # (C,)
        
        # Compute mean and variance only on valid pixels
        masked_x = x * mask.bool()
        mean = masked_x.sum(dim=(0, 2, 3)) / valid_count.clamp(min=1)  # (C,)
        
        var = ((x - mean.view(1, C, 1, 1)) ** 2 * mask.bool()).sum(dim=(0, 2, 3)) / valid_count.clamp(min=1)
        
        # Update running statistics only during training
        if self.training:
            self.running_mean.mul_(1 - self.momentum).add_(mean.detach(), alpha=self.momentum)
            self.running_var.mul_(1 - self.momentum).add_(var.detach(), alpha=self.momentum)
        
        # Normalize
        mean = mean.view(1, C, 1, 1)
        var = var.view(1, C, 1, 1)
        weight = self.weight.view(1, C, 1, 1)
        bias = self.bias.view(1, C, 1, 1)
        
        x_norm = (x - mean) / torch.sqrt(var + self.eps) * weight + bias
        
        # Only apply normalization to valid pixels
        return torch.where(mask, x_norm, x)
    
# register normalization function here
REGISTERED_NORM_DICT: dict[str, type] = {
    "bn2d": BatchNorm2d,
    "ln": nn.LayerNorm,
    "ln2d": LayerNorm2d,
}


def build_norm(name="bn2d", num_features=None, **kwargs) -> nn.Module or None:
    if name in ["ln", "ln2d"]:
        kwargs["normalized_shape"] = num_features
    else:
        kwargs["num_features"] = num_features
    if name in REGISTERED_NORM_DICT:
        norm_cls = REGISTERED_NORM_DICT[name]
        args = build_kwargs_from_config(kwargs, norm_cls)
        return norm_cls(**args)
    else:
        return None


def reset_bn(
    model: nn.Module,
    data_loader: list,
    sync=True,
    progress_bar=False,
) -> None:
    import copy

    import torch.nn.functional as F
    import torchpack.distributed as dist
    from tqdm import tqdm

    from efficientvit.apps.utils import AverageMeter, sync_tensor
    from efficientvit.models.utils import get_device, list_join

    bn_mean = {}
    bn_var = {}

    tmp_model = copy.deepcopy(model)
    for name, m in tmp_model.named_modules():
        if isinstance(m, _BatchNorm):
            bn_mean[name] = AverageMeter(is_distributed=False)
            bn_var[name] = AverageMeter(is_distributed=False)

            def new_forward(bn, mean_est, var_est):
                def lambda_forward(x):
                    x = x.contiguous()
                    if sync:
                        batch_mean = x.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)  # 1, C, 1, 1
                        batch_mean = sync_tensor(batch_mean, reduce="cat")
                        batch_mean = torch.mean(batch_mean, dim=0, keepdim=True)

                        batch_var = (x - batch_mean) * (x - batch_mean)
                        batch_var = batch_var.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)
                        batch_var = sync_tensor(batch_var, reduce="cat")
                        batch_var = torch.mean(batch_var, dim=0, keepdim=True)
                    else:
                        batch_mean = x.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)  # 1, C, 1, 1
                        batch_var = (x - batch_mean) * (x - batch_mean)
                        batch_var = batch_var.mean(0, keepdim=True).mean(2, keepdim=True).mean(3, keepdim=True)

                    batch_mean = torch.squeeze(batch_mean)
                    batch_var = torch.squeeze(batch_var)

                    mean_est.update(batch_mean.data, x.size(0))
                    var_est.update(batch_var.data, x.size(0))

                    # bn forward using calculated mean & var
                    _feature_dim = batch_mean.shape[0]
                    return F.batch_norm(
                        x,
                        batch_mean,
                        batch_var,
                        bn.weight[:_feature_dim],
                        bn.bias[:_feature_dim],
                        False,
                        0.0,
                        bn.eps,
                    )

                return lambda_forward

            m.forward = new_forward(m, bn_mean[name], bn_var[name])

    # skip if there is no batch normalization layers in the network
    if len(bn_mean) == 0:
        return

    tmp_model.eval()
    with torch.no_grad():
        with tqdm(total=len(data_loader), desc="reset bn", disable=not progress_bar or not dist.is_master()) as t:
            for images in data_loader:
                images = images.to(get_device(tmp_model))
                tmp_model(images)
                t.set_postfix(
                    {
                        "bs": images.size(0),
                        "res": list_join(images.shape[-2:], "x"),
                    }
                )
                t.update()

    for name, m in model.named_modules():
        if name in bn_mean and bn_mean[name].count > 0:
            feature_dim = bn_mean[name].avg.size(0)
            assert isinstance(m, _BatchNorm)
            m.running_mean.data[:feature_dim].copy_(bn_mean[name].avg)
            m.running_var.data[:feature_dim].copy_(bn_var[name].avg)


def set_norm_eps(model: nn.Module, eps: float or None = None) -> None:
    for m in model.modules():
        if isinstance(m, (nn.GroupNorm, nn.LayerNorm, _BatchNorm)):
            if eps is not None:
                m.eps = eps
