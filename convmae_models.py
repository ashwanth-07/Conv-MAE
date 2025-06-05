# Modifications Copyright © 2025 Advanced Micro Devices, Inc. All rights reserved.
#

import torch
import torch.nn as nn
import torch.nn.functional as F


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, dropout=0.0, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = norm_layer(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(int(dim * mlp_ratio), dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class PositionalEncoding2D(nn.Module):
    """2D sine–cosine positional encoding."""
    def __init__(self, channels, height, width):
        super().__init__()
        self.channels = channels
        self.height = height
        self.width = width

        pe = self._get_2d_sincos_pos_embed(channels, height, width)
        self.register_buffer('pos_embed', pe)

    def _get_2d_sincos_pos_embed(self, embed_dim, grid_h, grid_w):
        assert embed_dim % 2 == 0, "Embedding dimension must be even"
        
        emb_h = self._get_1d_sincos_pos_embed(embed_dim // 2, grid_h)
        emb_w = self._get_1d_sincos_pos_embed(embed_dim // 2, grid_w)
        
        emb_h = emb_h.unsqueeze(1).repeat(1, grid_w, 1)
        emb_w = emb_w.unsqueeze(0).repeat(grid_h, 1, 1)
        
        emb = torch.cat([emb_h, emb_w], dim=-1)
        return emb.view(grid_h * grid_w, embed_dim)

    def _get_1d_sincos_pos_embed(self, embed_dim, length):
        omega = torch.arange(embed_dim // 2, dtype=torch.float32)
        omega /= embed_dim / 2.0
        omega = 1.0 / (10000 ** omega)
        
        pos = torch.arange(length, dtype=torch.float32)
        out = torch.einsum('i,j->ij', pos, omega)
        
        emb_sin = torch.sin(out)
        emb_cos = torch.cos(out)
        emb = torch.cat([emb_sin, emb_cos], dim=1)
        
        return emb

    def forward(self, tokens):
        B, N, D = tokens.shape
        assert N == self.height * self.width, f"Expected {self.height * self.width} tokens, got {N}"
        assert D == self.channels, f"Expected {self.channels} channels, got {D}"
        
        pe = self.pos_embed.unsqueeze(0).expand(B, -1, -1)
        return tokens + pe.to(tokens.device)


class ConvMAEDecoder(nn.Module):
    """
    Extended ConvMAE multi-scale decoder that projects encoder features from 
    H/4, H/8, H/16, H/32 to H/16 resolution, fuses via concatenation + MLP,
    applies transformer processing, and reconstructs the full image.
    """
    def __init__(
        self,
        encoder_dims: list,      # [C1, C2, C3, C4] for H/4, H/8, H/16, H/32
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

        assert len(encoder_dims) == 4, f"Expected 4 encoder dimensions, got {len(encoder_dims)}"
        assert decoder_dim % decoder_num_heads == 0, f"decoder_dim ({decoder_dim}) must be divisible by num_heads ({decoder_num_heads})"

        self.encoder_dims = encoder_dims
        self.decoder_dim = decoder_dim
        self.patch_size = patch_size
        self.in_channels = in_channels

        self.proj_e1 = nn.Conv2d(encoder_dims[0], decoder_dim, kernel_size=4, stride=4)
        self.proj_e2 = nn.Conv2d(encoder_dims[1], decoder_dim, kernel_size=2, stride=2)
        self.proj_e3 = nn.Conv2d(encoder_dims[2], decoder_dim, kernel_size=1, stride=1)
        self.proj_e4 = nn.Conv2d(encoder_dims[3], decoder_dim, kernel_size=1, stride=1)

        self.feature_norm = norm_layer(decoder_dim)

        self.multi_scale_fusion = nn.Sequential(
            nn.Linear(decoder_dim * 4, decoder_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(decoder_dim * 2, decoder_dim),
            nn.Dropout(dropout)
        )

        self.mask_token = nn.Parameter(torch.zeros(1, decoder_dim))
        self.pos_embed = None

        self.decoder_blocks = nn.ModuleList([
            TransformerBlock(
                dim=decoder_dim,
                num_heads=decoder_num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                norm_layer=norm_layer
            )
            for _ in range(decoder_depth)
        ])

        self.decoder_norm = norm_layer(decoder_dim)
        self.decoder_pred = nn.Linear(decoder_dim, patch_size**2 * in_channels, bias=True)

        self.initialize_weights()

    def initialize_weights(self):
        nn.init.normal_(self.mask_token, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _init_pos_embed(self, H: int, W: int):
        if (self.pos_embed is None or 
            self.pos_embed.height != H or 
            self.pos_embed.width != W):
            self.pos_embed = PositionalEncoding2D(
                channels=self.decoder_dim,
                height=H,
                width=W
            )

    def forward(self, multi_scale_features: list, mask: torch.Tensor) -> torch.Tensor:
        assert len(multi_scale_features) == 4, f"Expected 4 feature maps, got {len(multi_scale_features)}"
        
        B = multi_scale_features[0].shape[0]
        _, _, H16, W16 = multi_scale_features[2].shape
        
        expected_mask_shape = (B, 1, H16, W16)
        if mask.shape != expected_mask_shape:
            mask = F.interpolate(mask, size=(H16, W16), mode='nearest')

        feat1 = self.proj_e1(multi_scale_features[0])
        feat2 = self.proj_e2(multi_scale_features[1])
        feat3 = self.proj_e3(multi_scale_features[2])

        feat4_coarse = self.proj_e4(multi_scale_features[3])
        feat4 = F.interpolate(feat4_coarse, size=(H16, W16),
                              mode='bilinear', align_corners=False)

        def flatten_and_norm(x):
            x_flat = x.flatten(2).transpose(1, 2)
            return self.feature_norm(x_flat)

        p1 = flatten_and_norm(feat1)
        p2 = flatten_and_norm(feat2)
        p3 = flatten_and_norm(feat3)
        p4 = flatten_and_norm(feat4)

        concatenated = torch.cat([p1, p2, p3, p4], dim=-1)
        fused = self.multi_scale_fusion(concatenated)

        mask_flat = mask.flatten(2).transpose(1, 2).squeeze(-1)

        decoder_tokens = torch.zeros_like(fused)
        for b in range(B):
            vis_idx = mask_flat[b].bool()
            decoder_tokens[b, vis_idx] = fused[b, vis_idx]
            masked_idx = ~vis_idx
            if masked_idx.sum() > 0:
                decoder_tokens[b, masked_idx] = self.mask_token.expand(
                    masked_idx.sum(), self.decoder_dim
                )

        self._init_pos_embed(H16, W16)
        decoder_tokens = self.pos_embed(decoder_tokens)

        x = decoder_tokens
        for blk in self.decoder_blocks:
            x = blk(x)

        x = self.decoder_norm(x)
        pred = self.decoder_pred(x)

        pred = pred.view(B, H16, W16, self.patch_size, self.patch_size, self.in_channels)
        pred = pred.permute(0, 5, 1, 3, 2, 4)
        pred = pred.reshape(B, self.in_channels, H16 * self.patch_size, W16 * self.patch_size)
        
        return pred

    def forward_loss(
        self, 
        imgs: torch.Tensor, 
        pred: torch.Tensor, 
        mask: torch.Tensor,
        norm_pix_loss: bool = True
    ) -> torch.Tensor:
        if mask.shape[-2:] != imgs.shape[-2:]:
            mask_resized = F.interpolate(mask.float(), size=imgs.shape[-2:], mode='nearest')
        else:
            mask_resized = mask.float()
        
        loss_mask = 1 - mask_resized
        
        if norm_pix_loss:
            mean = imgs.mean(dim=(2, 3), keepdim=True)
            var = imgs.var(dim=(2, 3), keepdim=True)
            target = (imgs - mean) / (var + 1e-6).sqrt()
        else:
            target = imgs
        
        loss = (pred - target).pow(2)
        loss = loss.mean(dim=1, keepdim=True)
        loss = (loss * loss_mask).sum() / (loss_mask.sum() + 1e-8)
        
        return loss

# Example usage function
def create_convmae_decoder_for_efficientvit(
    backbone_name: str = "b2",
    decoder_dim: int = 512,
    decoder_depth: int = 8,
    decoder_num_heads: int = 16,
    patch_size: int = 16
) -> ConvMAEDecoder:
    """
    Create ConvMAE decoder configured for specific EfficientViT backbone
    """
    # Define encoder dimensions for different EfficientViT models
    # Uses 4 stages: [stage1, stage2, stage3, stage4] corresponding to [H/4, H/8, H/16, H/32]
    backbone_configs = {
        "b0": [16, 32, 64, 128],      # stages 1, 2, 3, 4
        "b1": [32, 64, 128, 256],     # stages 1, 2, 3, 4
        "b2": [48, 96, 192, 384],     # stages 1, 2, 3, 4
        "b3": [64, 128, 256, 512],    # stages 1, 2, 3, 4
        "l0": [64, 128, 256, 512],    # stages 1, 2, 3, 4
        "l1": [64, 128, 256, 512],    # stages 1, 2, 3, 4
        "l2": [64, 128, 256, 512],    # stages 1, 2, 3, 4
        "l3": [128, 256, 512, 1024],  # stages 1, 2, 3, 4
    }
    
    if backbone_name not in backbone_configs:
        raise ValueError(f"Unsupported backbone: {backbone_name}")
    
    encoder_dims = backbone_configs[backbone_name]
    
    decoder = ConvMAEDecoder(
        encoder_dims=encoder_dims,
        decoder_dim=decoder_dim,
        decoder_depth=decoder_depth,
        decoder_num_heads=decoder_num_heads,
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
        decoder_num_heads: int = 16,
        patch_size: int = 16
    ):
        super().__init__()
        
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
            decoder_num_heads=decoder_num_heads,
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
        
        # Extract 4 stages for ConvMAEDecoder
        multi_scale_features = [
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