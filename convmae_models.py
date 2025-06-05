# Modifications Copyright © 2025 Advanced Micro Devices, Inc. All rights reserved.
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional
import math

class PositionalEncoding2D(nn.Module):
    """2D positional encoding for transformer decoder"""
    def __init__(self, channels: int, height: int, width: int):
        super().__init__()
        self.channels = channels
        self.height = height
        self.width = width
        
        # Create positional encoding
        pos_embed = self._get_2d_sincos_pos_embed(channels, height, width)
        self.register_buffer('pos_embed', pos_embed)
    
    def _get_2d_sincos_pos_embed(self, embed_dim: int, grid_h: int, grid_w: int):
        """2D sin-cos position embedding based on MAE implementation"""
        grid_h_pos = torch.arange(grid_h, dtype=torch.float32)
        grid_w_pos = torch.arange(grid_w, dtype=torch.float32)
        grid = torch.meshgrid(grid_w_pos, grid_h_pos, indexing='xy')
        grid = torch.stack(grid, dim=0)  # 2, H, W
        
        pos_embed = self._get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
        return pos_embed.unsqueeze(0)  # 1, H*W, C
    
    def _get_2d_sincos_pos_embed_from_grid(self, embed_dim: int, grid):
        assert embed_dim % 2 == 0
        
        # Use half of dimensions to encode grid_h
        emb_h = self._get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # H*W, C/2
        emb_w = self._get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # H*W, C/2
        
        emb = torch.cat([emb_h, emb_w], dim=1)  # H*W, C
        return emb
    
    def _get_1d_sincos_pos_embed_from_grid(self, embed_dim: int, pos):
        """
        embed_dim: output dimension for each position
        pos: a list of positions to be encoded: size (M,)
        out: (M, D)
        """
        assert embed_dim % 2 == 0
        omega = torch.arange(embed_dim // 2, dtype=torch.float32)
        omega /= embed_dim / 2.
        omega = 1. / 10000**omega  # (D/2,)
        
        pos = pos.reshape(-1)  # (M,)
        out = torch.einsum('m,d->md', pos, omega)  # (M, D/2), outer product
        
        emb_sin = torch.sin(out)  # (M, D/2)
        emb_cos = torch.cos(out)  # (M, D/2)
        
        emb = torch.cat([emb_sin, emb_cos], dim=1)  # (M, D)
        return emb
    
    def forward(self, x: torch.Tensor):
        # x: B, N, C
        return x + self.pos_embed.to(x.device)


class TransformerBlock(nn.Module):
    """Transformer block for decoder"""
    def __init__(
        self, 
        dim: int, 
        num_heads: int = 8, 
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        # Self-attention
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm, attn_mask=attn_mask)
        x = x + attn_out
        
        # MLP
        x = x + self.mlp(self.norm2(x))
        return x


class ConvMAEMultiScaleDecoder(nn.Module):
    """
    ConvMAE-style multiscale decoder that fuses all 5 encoder stages:
      Stage0: H/2 → C₀
      Stage1: H/4 → C₁  
      Stage2: H/8 → C₂
      Stage3: H/16 → C₃
      Stage4: H/32 → C₄
    """
    def __init__(
        self,
        encoder_dims: List[int],  # Now length=5, e.g. [C₀, C₁, C₂, C₃, C₄]
        decoder_dim: int = 512,
        decoder_depth: int = 8,
        decoder_num_heads: int = 16,
        mlp_ratio: float = 4.0,
        patch_size: int = 16,
        in_channels: int = 3,
        norm_layer: nn.Module = nn.LayerNorm,
        dropout: float = 0.0
    ):
        super().__init__()
        
        self.encoder_dims = encoder_dims
        self.decoder_dim = decoder_dim
        self.patch_size = patch_size
        self.in_channels = in_channels
        
        # Downsample each of the five stages to H/32 via strided conv:
        # Stage0 @H/2  → kernel=16, stride=16 → H/32
        # Stage1 @H/4  → kernel=8,  stride=8  → H/32
        # Stage2 @H/8  → kernel=4,  stride=4  → H/32
        # Stage3 @H/16 → kernel=2,  stride=2  → H/32
        # Stage4 @H/32 → kernel=1,  stride=1  → H/32
        downsample_ratios = [16, 8, 4, 2, 1]  # One entry per encoder_dims
        
        self.stage_projections = nn.ModuleList()
        for dim, ratio in zip(encoder_dims, downsample_ratios):
            if ratio > 1:
                # Just the conv projection, no normalization here
                proj = nn.Conv2d(dim, decoder_dim, kernel_size=ratio, stride=ratio)
            else:
                # ratio==1 means "already H/32", so just project channels
                proj = nn.Conv2d(dim, decoder_dim, kernel_size=1, stride=1)
            self.stage_projections.append(proj)
        
        # Apply LayerNorm after flattening to tokens
        self.feature_norm = nn.LayerNorm(decoder_dim)
        
        # Fuse all five projected features by concatenating their feature vectors
        self.multi_scale_fusion = nn.Linear(decoder_dim * len(encoder_dims), decoder_dim)
        
        # Mask token and positional embedding (will be sized to H/32 at runtime)
        self.mask_token = nn.Parameter(torch.zeros(1, decoder_dim))  # Fix: 2D tensor, not 3D
        self.pos_embed = None  # Instantiated lazily in _init_pos_embed
        
        # Transformer decoder blocks
        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(
                dim=decoder_dim,
                num_heads=decoder_num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                norm_layer=norm_layer
            ) for _ in range(decoder_depth)
        ])
        
        self.decoder_norm = norm_layer(decoder_dim)
        self.decoder_pred = nn.Linear(decoder_dim, patch_size**2 * in_channels, bias=True)
        
        self.initialize_weights()
    
    def initialize_weights(self):
        """Initialize mask token and linear/conv weights as in MAE."""
        torch.nn.init.normal_(self.mask_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _init_pos_embed(self, H: int, W: int):
        """Create a 2D sin-cos embedding of size (H*W × decoder_dim)."""
        if self.pos_embed is None or self.pos_embed.height != H or self.pos_embed.width != W:
            self.pos_embed = PositionalEncoding2D(
                channels=self.decoder_dim,
                height=H,
                width=W
            )
    
    def forward(
        self, 
        multi_scale_features: List[torch.Tensor],  # Length=5: [feat_H/2, feat_H/4, feat_H/8, feat_H/16, feat_H/32]
        mask: torch.Tensor,  # Binary mask at whatever resolution (we'll downsample it to H/32)
        return_token_num: Optional[int] = None
    ) -> torch.Tensor:
        """
        Args:
            multi_scale_features: List of 5 feature tensors from encoder stages
            mask: Binary mask tensor (1=visible, 0=masked)
            return_token_num: Number of tokens to return (for efficiency)
        
        Returns:
            Reconstructed image patches
        """
        B = multi_scale_features[0].shape[0]
        
        # 1) Project each stage to (B, decoder_dim, H/32, W/32), then flatten → (B, N, decoder_dim)
        projected_features = []
        for feat, proj in zip(multi_scale_features, self.stage_projections):
            # feat might be (B, Cᵢ, Hᵢ, Wᵢ) with Hᵢ ∈ {H/2, H/4, H/8, H/16, H/32}
            proj_feat = proj(feat)                   # → (B, decoder_dim, H/32, W/32)
            proj_feat = proj_feat.flatten(2)         # → (B, decoder_dim, (H/32)*(W/32))
            proj_feat = proj_feat.transpose(1, 2)    # → (B, N, decoder_dim), where N = (H/32)*(W/32)
            proj_feat = self.feature_norm(proj_feat) # Apply LayerNorm to tokens
            projected_features.append(proj_feat)
        
        # 2) Concatenate on the feature-dimension (depth) and fuse back to decoder_dim
        #    concatenated shape: (B, N, decoder_dim * 5)
        fused = torch.cat(projected_features, dim=-1)
        fused = self.multi_scale_fusion(fused)  # → (B, N, decoder_dim)
        
        # 3) Prepare binary mask at H/32 resolution
        #    If `mask` was made at some coarser grid, just interpolate it down to (H/32, W/32)
        _, _, H32, W32 = multi_scale_features[-1].shape  # H32 = H/32, etc.
        if mask.shape[-2:] != (H32, W32):
            mask_resized = F.interpolate(mask.float(), size=(H32, W32), mode='nearest')
        else:
            mask_resized = mask.float()
        mask_flat = mask_resized.flatten(2).transpose(1, 2).squeeze(-1)  # → (B, N) where N = H32*W32
        
        # 4) Build decoder tokens: visible=fused, masked=mask_token
        decoder_tokens = torch.zeros(B, H32 * W32, self.decoder_dim, device=fused.device)
        for b in range(B):
            vis_idx = mask_flat[b].bool()
            decoder_tokens[b, vis_idx] = fused[b, vis_idx]
            masked_idx = ~vis_idx
            if masked_idx.sum() > 0:
                # Fix: mask_token is now shape (1, D), so expand to (num_masked, D)
                decoder_tokens[b, masked_idx] = self.mask_token.expand(masked_idx.sum(), self.decoder_dim)
        
        # 5) Add positional embedding (H/32 × W/32)
        self._init_pos_embed(H32, W32)
        decoder_tokens = self.pos_embed(decoder_tokens)  # (B, N, decoder_dim)
        
        # 6) Run through Transformer decoder blocks
        for blk in self.decoder_blocks:
            decoder_tokens = blk(decoder_tokens)
        
        # 7) Final norm + prediction head → (B, N, patch_size^2 * in_channels)
        decoder_tokens = self.decoder_norm(decoder_tokens)
        pred = self.decoder_pred(decoder_tokens)
        
        # 8) Reshape to pixels: (B, C, H, W)
        #    Here N = H32*W32, so pred.view → (B, H32, W32, p, p, C)
        pred = pred.view(B, H32, W32, self.patch_size, self.patch_size, self.in_channels)
        pred = pred.permute(0, 5, 1, 3, 2, 4)  # → (B, C, H32, p, W32, p)
        pred = pred.reshape(B, self.in_channels, H32 * self.patch_size, W32 * self.patch_size)
        return pred
    
    def forward_loss(
        self, 
        imgs: torch.Tensor, 
        pred: torch.Tensor, 
        mask: torch.Tensor,
        norm_pix_loss: bool = True
    ) -> torch.Tensor:
        """
        Calculate reconstruction loss only on masked patches
        
        Args:
            imgs: Original images (B, C, H, W)
            pred: Predicted images (B, C, H, W) 
            mask: Binary mask where 1=visible, 0=masked
            norm_pix_loss: Whether to normalize pixel loss
        """
        # Resize mask to match image resolution if needed
        if mask.shape[-2:] != imgs.shape[-2:]:
            mask_resized = F.interpolate(mask.float(), size=imgs.shape[-2:], mode='nearest')
        else:
            mask_resized = mask.float()
        
        # We want to compute loss only on masked regions (where mask == 0)
        loss_mask = 1 - mask_resized  # Invert mask: 1 = masked, 0 = visible
        
        if norm_pix_loss:
            # Normalize target patches
            mean = imgs.mean(dim=(2, 3), keepdim=True)
            var = imgs.var(dim=(2, 3), keepdim=True)
            target = (imgs - mean) / (var + 1e-6).sqrt()
        else:
            target = imgs
        
        # Compute L2 loss
        loss = (pred - target).pow(2)
        loss = loss.mean(dim=1, keepdim=True)  # Average over channels → (B,1,H,W)
        
        # Apply mask (only compute loss on masked regions)
        loss = (loss * loss_mask).sum() / loss_mask.sum()  # Mean loss on masked patches
        
        return loss


# Example usage function
def create_convmae_decoder_for_efficientvit(
    backbone_name: str = "b2",
    decoder_dim: int = 512,
    decoder_depth: int = 8,
    patch_size: int = 16
) -> ConvMAEMultiScaleDecoder:
    """
    Create ConvMAE decoder configured for specific EfficientViT backbone
    """
    # Define encoder dimensions for different EfficientViT models
    # Now includes all 5 stages: [stage0, stage1, stage2, stage3, stage4]
    backbone_configs = {
        "b0": [8, 16, 32, 64, 128],      # stages 0, 1, 2, 3, 4
        "b1": [16, 32, 64, 128, 256],    # stages 0, 1, 2, 3, 4
        "b2": [24, 48, 96, 192, 384],    # stages 0, 1, 2, 3, 4
        "b3": [32, 64, 128, 256, 512],   # stages 0, 1, 2, 3, 4
        "l0": [32, 64, 128, 256, 512],   # stages 0, 1, 2, 3, 4
        "l1": [32, 64, 128, 256, 512],   # stages 0, 1, 2, 3, 4
        "l2": [32, 64, 128, 256, 512],   # stages 0, 1, 2, 3, 4
        "l3": [64, 128, 256, 512, 1024], # stages 0, 1, 2, 3, 4
    }
    
    if backbone_name not in backbone_configs:
        raise ValueError(f"Unsupported backbone: {backbone_name}")
    
    encoder_dims = backbone_configs[backbone_name]
    
    decoder = ConvMAEMultiScaleDecoder(
        encoder_dims=encoder_dims,
        decoder_dim=decoder_dim,
        decoder_depth=decoder_depth,
        patch_size=patch_size
    )
    
    return decoder


# Complete usage example with EfficientViT backbone
class ConvMAEPretrainer(nn.Module):
    """Complete ConvMAE pretrainer with EfficientViT backbone"""
    
    def __init__(
        self,
        backbone_name: str = "b2",
        mask_ratio: float = 0.75,
        decoder_dim: int = 512,
        decoder_depth: int = 8,
        patch_size: int = 16
    ):
        super().__init__()
        
        # Import your backbone creation function
        # from efficientvit.models.efficientvit.backbone import efficientvit_backbone_b2
        
        # Create backbone with masking enabled
        if backbone_name == "b0":
            from efficientvit.models.efficientvit.backbone import efficientvit_backbone_b0
            self.backbone = efficientvit_backbone_b0(mask_ratio=mask_ratio)
        elif backbone_name == "b1":
            from efficientvit.models.efficientvit.backbone import efficientvit_backbone_b1
            self.backbone = efficientvit_backbone_b1(mask_ratio=mask_ratio)
        elif backbone_name == "b2":
            from efficientvit.models.efficientvit.backbone import efficientvit_backbone_b2
            self.backbone = efficientvit_backbone_b2(mask_ratio=mask_ratio)
        elif backbone_name == "b3":
            from efficientvit.models.efficientvit.backbone import efficientvit_backbone_b3
            self.backbone = efficientvit_backbone_b3(mask_ratio=mask_ratio)
        else:
            raise ValueError(f"Unsupported backbone: {backbone_name}")
        
        # Create decoder
        self.decoder = create_convmae_decoder_for_efficientvit(
            backbone_name=backbone_name,
            decoder_dim=decoder_dim,
            decoder_depth=decoder_depth,
            patch_size=patch_size
        )
        
        self.mask_ratio = mask_ratio
    
    def forward(self, images: torch.Tensor):
        """
        Forward pass for pretraining
        
        Args:
            images: Input images (B, 3, H, W)
        
        Returns:
            pred: Reconstructed images (B, 3, H, W)
            loss: Reconstruction loss
            mask: Applied mask
        """
        # Get multi-scale features from backbone
        backbone_output = self.backbone(images)
        
        # Extract all 5 stages
        multi_scale_features = [
            backbone_output["stage0"],  # H/2 resolution
            backbone_output["stage1"],  # H/4 resolution
            backbone_output["stage2"],  # H/8 resolution
            backbone_output["stage3"],  # H/16 resolution
            backbone_output["stage4"],  # H/32 resolution
        ]
        
        # Generate mask (this should be done inside the backbone)
        mask = self.backbone.generate_mask(images)
        
        # Decode
        pred = self.decoder(multi_scale_features, mask)
        
        # Compute loss
        loss = self.decoder.forward_loss(images, pred, mask)
        
        return pred, loss, mask