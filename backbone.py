import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Union

# Import official EfficientViT components
from efficientvit.models.efficientvit.backbone import EfficientViTBackbone, EfficientViTLargeBackbone


class MaskGenerator:
    """Generates random block masks for ConvMAE training."""
    
    def __init__(self, mask_ratio: float = 0.75):
        self.mask_ratio = mask_ratio
    
    def create_mask(self, batch_size: int, height: int, width: int, device: torch.device) -> torch.Tensor:
        """Create a random block mask.
        
        Returns:
            mask: (B, 1, H, W) tensor where True = masked, False = keep
        """
        num_patches = height * width
        num_keep = int((1 - self.mask_ratio) * num_patches)
        
        # Create mask for each batch item
        masks = []
        for _ in range(batch_size):
            # Start with all patches masked
            mask = torch.ones(num_patches, device=device, dtype=torch.bool)
            # Randomly select patches to keep
            keep_indices = torch.randperm(num_patches, device=device)[:num_keep]
            mask[keep_indices] = False  # False = keep
            masks.append(mask.view(1, height, width))
        
        return torch.stack(masks, dim=0).unsqueeze(1)  # (B, 1, H, W)
    
    def resize_mask(self, mask: torch.Tensor, target_size: tuple) -> torch.Tensor:
        """Resize mask to target spatial dimensions."""
        if mask.shape[2:] == target_size:
            return mask
        return F.interpolate(mask.float(), size=target_size, mode='nearest').bool()


class MaskedConvLayer(nn.Module):
    """Applies masking to any convolution layer."""
    
    def __init__(self, conv_layer: nn.Module):
        super().__init__()
        self.conv_layer = conv_layer
    
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if mask is None:
            return self.conv_layer(x)
        
        # Resize mask to match input if needed
        if mask.shape[2:] != x.shape[2:]:
            mask = F.interpolate(mask.float(), size=x.shape[2:], mode='nearest').bool()
        
        # Apply mask: zero out masked regions
        masked_input = x * (~mask).float()
        
        # Forward through convolution
        output = self.conv_layer(masked_input)
        
        # Apply mask to output too
        output_mask = F.interpolate(mask.float(), size=output.shape[2:], mode='nearest').bool()
        output = output * (~output_mask).float()
        
        return output


class ConvMAEEncoder(nn.Module):
    """ConvMAE encoder that wraps EfficientViT backbone with masking."""
    
    def __init__(self, backbone: Union[EfficientViTBackbone, EfficientViTLargeBackbone], mask_ratio: float = 0.75):
        super().__init__()
        self.backbone = backbone
        self.mask_ratio = mask_ratio
        self.mask_generator = MaskGenerator(mask_ratio)
        self._current_mask = None
    
    def forward(self, x: torch.Tensor, return_features: bool = False) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size, _, height, width = x.shape
        
        # Generate mask during training
        if self.training and self.mask_ratio > 0:
            # Create base mask at a reasonable resolution (final feature map size)
            mask_h, mask_w = height // 16, width // 16  # Assuming 16x total downsampling
            self._current_mask = self.mask_generator.create_mask(batch_size, mask_h, mask_w, x.device)
        else:
            self._current_mask = None
        
        # Forward through backbone with selective masking
        features = self._forward_with_masking(x)
        
        if return_features:
            if self._current_mask is not None:
                features['mask'] = self._current_mask
            return features
        else:
            return features['final']
    
    def _forward_with_masking(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass applying masking selectively to safe layers."""
        features = {'input': x}
        current = x
        
        # Forward through input stem
        current = self._forward_stage(current, self.backbone.input_stem, 'stem')
        features['stem'] = current
        
        # Forward through each stage
        for i, stage in enumerate(self.backbone.stages):
            current = self._forward_stage(current, stage, f'stage_{i+1}')
            features[f'stage_{i+1}'] = current
        
        features['final'] = current
        return features
    
    def _forward_stage(self, x: torch.Tensor, stage: nn.Module, stage_name: str) -> torch.Tensor:
        """Forward through a stage, applying masking where appropriate."""
        if self._current_mask is None:
            return stage(x)
        
        current = x
        # Process each block in the stage
        for block in stage.op_list:
            if self._should_apply_masking(block):
                current = self._forward_with_mask(current, block)
            else:
                current = block(current)
        
        return current
    
    def _should_apply_masking(self, block: nn.Module) -> bool:
        """Determine if we should apply masking to this block.
        
        Only apply masking to pure convolution layers, not attention layers.
        """
        from efficientvit.models.nn import ConvLayer, MBConv, DSConv, ResidualBlock
        
        # Safe conv layers
        if isinstance(block, (ConvLayer, MBConv, DSConv)):
            return True
        
        # Residual blocks with conv main branch
        if isinstance(block, ResidualBlock) and hasattr(block, 'main'):
            return isinstance(block.main, (ConvLayer, MBConv, DSConv))
        
        # Don't mask attention layers (EfficientViTBlock)
        return False
    
    def _forward_with_mask(self, x: torch.Tensor, block: nn.Module) -> torch.Tensor:
        """Forward through a block with masking applied."""
        from efficientvit.models.nn import ResidualBlock
        
        # Get mask at current resolution
        current_mask = self.mask_generator.resize_mask(self._current_mask, x.shape[2:])
        
        if isinstance(block, ResidualBlock):
            return self._forward_residual_with_mask(x, block, current_mask)
        else:
            return MaskedConvLayer(block)(x, current_mask)
    
    def _forward_residual_with_mask(self, x: torch.Tensor, block: nn.Module, mask: torch.Tensor) -> torch.Tensor:
        """Forward through residual block with masking."""
        # Main branch with masking
        main_out = MaskedConvLayer(block.main)(x, mask) if block.main else x
        
        # Shortcut branch (no masking)
        shortcut_out = block.shortcut(x) if block.shortcut else x
        
        # Add residual connection if shapes match
        if main_out.shape == shortcut_out.shape:
            out = main_out + shortcut_out
        else:
            out = main_out
        
        # Post activation if present
        if hasattr(block, 'post_act') and block.post_act:
            out = block.post_act(out)
        
        return out


class ConvMAEDecoder(nn.Module):
    """Simple decoder for ConvMAE reconstruction."""
    
    def __init__(self, encoder_dim: int, embed_dim: int = 512, depth: int = 8):
        super().__init__()
        self.embed_dim = embed_dim
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Project encoder features to decoder dimension
        self.encoder_proj = nn.Linear(encoder_dim, embed_dim)
        
        # Transformer decoder
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=embed_dim * 4,
            dropout=0.0,
            activation='gelu',
            batch_first=True
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=depth)
        
        # Prediction head (predicts 16x16 RGB patches)
        self.pred_head = nn.Linear(embed_dim, 16 * 16 * 3)
        
        # Initialize mask token
        nn.init.trunc_normal_(self.mask_token, std=0.02)
    
    def forward(self, encoder_features: torch.Tensor) -> torch.Tensor:
        """Decode encoder features to reconstruct patches."""
        batch_size = encoder_features.shape[0]
        
        # Global average pool encoder features
        pooled = F.adaptive_avg_pool2d(encoder_features, 1).flatten(1)
        
        # Project to decoder dimension
        encoder_tokens = self.encoder_proj(pooled).unsqueeze(1)  # (B, 1, embed_dim)
        
        # Add mask tokens (196 = 14x14 patches for 224x224 images)
        mask_tokens = self.mask_token.expand(batch_size, 196, -1)
        decoder_input = torch.cat([encoder_tokens, mask_tokens], dim=1)
        
        # Decode
        decoded = self.decoder(decoder_input)
        
        # Predict patches (skip the first encoder token)
        predictions = self.pred_head(decoded[:, 1:, :])
        
        return predictions


class ConvMAE(nn.Module):
    """Complete ConvMAE model for self-supervised pretraining."""
    
    def __init__(
        self,
        backbone_fn,
        mask_ratio: float = 0.75,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        **backbone_kwargs
    ):
        super().__init__()
        
        # Create backbone and wrap with ConvMAE encoder
        backbone = backbone_fn(**backbone_kwargs)
        self.encoder = ConvMAEEncoder(backbone, mask_ratio)
        
        # Create decoder
        encoder_dim = backbone.width_list[-1]  # Final feature dimension
        self.decoder = ConvMAEDecoder(encoder_dim, decoder_embed_dim, decoder_depth)
    
    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for training."""
        # Encode with masking
        features = self.encoder(images, return_features=True)
        final_features = features['final']
        
        # Decode to patches
        predictions = self.decoder(final_features)
        
        # Convert images to patches for loss computation
        targets = self._images_to_patches(images)
        
        # Compute reconstruction loss
        loss = F.mse_loss(predictions, targets)
        
        return {
            'loss': loss,
            'predictions': predictions,
            'targets': targets,
            'features': features
        }
    
    def _images_to_patches(self, images: torch.Tensor, patch_size: int = 16) -> torch.Tensor:
        """Convert images to patches for loss computation."""
        batch_size, channels, height, width = images.shape
        
        # Ensure image dimensions are divisible by patch size
        assert height % patch_size == 0 and width % patch_size == 0
        
        h_patches = height // patch_size
        w_patches = width // patch_size
        
        # Reshape to patches: (B, C, H, W) -> (B, num_patches, patch_dim)
        patches = images.view(
            batch_size, channels, 
            h_patches, patch_size, 
            w_patches, patch_size
        )
        patches = patches.permute(0, 2, 4, 3, 5, 1)  # (B, h_patches, w_patches, patch_size, patch_size, C)
        patches = patches.reshape(batch_size, h_patches * w_patches, patch_size * patch_size * channels)
        
        return patches
    
    def save_backbone(self, filepath: str):
        """Save the encoder backbone for downstream tasks."""
        backbone_state = {
            'model_state_dict': self.encoder.backbone.state_dict(),
            'width_list': self.encoder.backbone.width_list,
            'architecture': type(self.encoder.backbone).__name__
        }
        torch.save(backbone_state, filepath)
        print(f"Backbone saved to {filepath}")


def load_pretrained_backbone(backbone_fn, checkpoint_path: str, **backbone_kwargs):
    """Load a pretrained ConvMAE backbone for downstream tasks."""
    backbone = backbone_fn(**backbone_kwargs)
    
    if checkpoint_path and torch.cuda.is_available():
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            state_dict = checkpoint.get('model_state_dict', checkpoint)
            
            missing, unexpected = backbone.load_state_dict(state_dict, strict=False)
            
            if missing:
                print(f"Missing keys: {len(missing)}")
            if unexpected:
                print(f"Unexpected keys: {len(unexpected)}")
                
            print("ConvMAE weights loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load checkpoint {checkpoint_path}: {e}")
            print("Using random initialization")
    
    return backbone


# Factory function
def convmae_efficientvit_b2(**kwargs):
    """Create ConvMAE model with EfficientViT-B2 backbone."""
    from efficientvit.models.efficientvit.backbone import efficientvit_backbone_b2
    return ConvMAE(backbone_fn=efficientvit_backbone_b2, **kwargs)


# Example usage and testing
if __name__ == "__main__":
    print("Testing Simplified ConvMAE implementation...")
    
    # Test without masking
    print("\n1. Testing without masking...")
    model = convmae_efficientvit_b2(mask_ratio=0.0)
    model.eval()
    
    x = torch.randn(2, 3, 224, 224)
    
    with torch.no_grad():
        output = model(x)
        print(f"✓ Forward pass successful! Loss: {output['loss'].item():.4f}")
    
    # Test with masking
    print("\n2. Testing with masking...")
    model = convmae_efficientvit_b2(mask_ratio=0.75)
    model.train()
    
    try:
        output = model(x)
        print(f"✓ Masked forward pass successful! Loss: {output['loss'].item():.4f}")
        
        # Save backbone
        model.save_backbone('simple_convmae_b2.pth')
        print("✓ Backbone saved successfully!")
        
    except Exception as e:
        print(f"✗ Error in masked forward pass: {e}")
        import traceback
        traceback.print_exc()
    
    # Test loading backbone
    print("\n3. Testing backbone loading...")
    try:
        from efficientvit.models.efficientvit.backbone import efficientvit_backbone_b2
        
        backbone = load_pretrained_backbone(
            efficientvit_backbone_b2,
            'simple_convmae_b2.pth'
        )
        
        with torch.no_grad():
            features = backbone(x)
            print(f"✓ Loaded backbone forward pass successful!")
            print(f"  Features shape: {features['stage_final'].shape}")
            
    except Exception as e:
        print(f"✗ Error loading backbone: {e}")
    
    print("\nSimplified ConvMAE test completed!")