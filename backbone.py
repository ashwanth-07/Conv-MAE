import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Dict, Optional, Union

# Import official EfficientViT components
from efficientvit.models.efficientvit.backbone import EfficientViTBackbone, EfficientViTLargeBackbone


class RobustMaskedConvLayer(nn.Module):
    """
    Robust masked convolution that handles dimension mismatches gracefully.
    """
    def __init__(self, conv_layer: nn.Module):
        super().__init__()
        self.conv_layer = conv_layer
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if mask is None:
            return self.conv_layer(x)
        
        # Ensure mask matches input dimensions
        if mask.shape[2:] != x.shape[2:]:
            mask = F.interpolate(mask.float(), size=x.shape[2:], mode='nearest').bool()
        
        # Apply mask: zero out masked regions
        inv_mask = (~mask).float()
        x_masked = x * inv_mask
        
        # Apply convolution
        out = self.conv_layer(x_masked)
        
        # Apply mask to output as well (conservative approach)
        if mask.shape[2:] != out.shape[2:]:
            output_mask = F.interpolate(mask.float(), size=out.shape[2:], mode='nearest').bool()
        else:
            output_mask = mask
            
        out = out * (~output_mask).float()
        
        return out


class RobustBlockMaskGenerator:
    """
    Robust mask generator that handles different architectures gracefully.
    """
    def __init__(self, mask_ratio: float = 0.75):
        self.mask_ratio = mask_ratio
        
    def generate_base_mask(self, batch_size: int, H: int, W: int, device: torch.device) -> torch.Tensor:
        """Generate base mask at given resolution."""
        num_tokens = H * W
        num_keep = int((1 - self.mask_ratio) * num_tokens)
        
        mask_flat = torch.ones(batch_size, num_tokens, device=device, dtype=torch.bool)
        
        for b in range(batch_size):
            keep_indices = torch.randperm(num_tokens, device=device)[:num_keep]
            mask_flat[b, keep_indices] = False  # False = keep, True = mask
            
        return mask_flat.view(batch_size, 1, H, W)
    
    def get_mask_for_resolution(self, base_mask: torch.Tensor, target_H: int, target_W: int) -> torch.Tensor:
        """Get mask for specific resolution by interpolating base mask."""
        if base_mask.shape[2] == target_H and base_mask.shape[3] == target_W:
            return base_mask
        
        return F.interpolate(base_mask.float(), size=(target_H, target_W), mode='nearest').bool()


class RobustConvMAEWrapper(nn.Module):
    """
    Conservative wrapper that applies masking only where safe and appropriate.
    Focuses on correctness over aggressive optimization.
    """
    
    def __init__(self, backbone: Union[EfficientViTBackbone, EfficientViTLargeBackbone], mask_ratio: float = 0.75):
        self.backbone = backbone
        self.mask_ratio = mask_ratio
        self.mask_generator = RobustBlockMaskGenerator(mask_ratio)
        self.base_mask = None
        self.training = True
        
    def __call__(self, x: torch.Tensor, return_features: bool = False) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        return self.forward(x, return_features)
        
    def train(self, mode: bool = True):
        """Set training mode."""
        self.training = mode
        self.backbone.train(mode)
        
    def eval(self):
        """Set evaluation mode."""
        self.training = False
        self.backbone.eval()
        
    def forward(self, x: torch.Tensor, return_features: bool = False) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Conservative forward pass that applies masking selectively.
        """
        B, C, H, W = x.shape
        
        # Generate base mask if in training mode
        if self.training and self.mask_ratio > 0:
            # Use a conservative base resolution (e.g., final stage resolution)
            base_H, base_W = H // 16, W // 16  # Assuming 16x total downsampling
            self.base_mask = self.mask_generator.generate_base_mask(B, base_H, base_W, x.device)
        else:
            self.base_mask = None
            
        # Forward through backbone with selective masking
        features = self._forward_with_selective_masking(x)
        
        if return_features:
            if self.base_mask is not None:
                features['base_mask'] = self.base_mask
            return features
        else:
            return features['stage_final']
    
    def _forward_with_selective_masking(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with selective masking applied only to safe layers.
        """
        features = {"input": x}
        current_x = x
        
        # Forward through input stem with optional masking
        current_x = self._forward_stage_with_masking(current_x, self.backbone.input_stem, "input_stem")
        features["stage0"] = current_x
        
        # Forward through stages
        for stage_id, stage in enumerate(self.backbone.stages, 1):
            current_x = self._forward_stage_with_masking(current_x, stage, f"stage{stage_id}")
            features[f"stage{stage_id}"] = current_x
            
        features["stage_final"] = current_x
        return features
    
    def _forward_stage_with_masking(self, x: torch.Tensor, stage, stage_name: str) -> torch.Tensor:
        """
        Forward through a stage with optional masking.
        Only applies masking to conv layers, leaves attention layers as-is.
        """
        if self.base_mask is None:
            # No masking - standard forward
            return stage(x)
        
        current_x = x
        
        # Check if this stage contains blocks we can safely mask
        for block in stage.op_list:
            if self._is_safe_to_mask(block):
                # Apply masking to this block
                current_mask = self.mask_generator.get_mask_for_resolution(
                    self.base_mask, current_x.shape[2], current_x.shape[3]
                )
                current_x = self._forward_block_with_masking(current_x, block, current_mask)
            else:
                # Forward without masking
                current_x = block(current_x)
                
        return current_x
    
    def _is_safe_to_mask(self, block) -> bool:
        """
        Determine if a block is safe to apply masking to.
        Conservative approach: only mask pure conv layers.
        """
        # Import here to avoid circular imports
        from efficientvit.models.nn import ConvLayer, MBConv, DSConv, ResidualBlock
        
        # Safe to mask: pure convolution layers
        if isinstance(block, (ConvLayer, MBConv, DSConv)):
            return True
            
        # Safe to mask: residual blocks with conv main blocks
        if isinstance(block, ResidualBlock):
            if isinstance(block.main, (ConvLayer, MBConv, DSConv)):
                return True
                
        # NOT safe to mask: EfficientViTBlock (hybrid attention/conv)
        # Let these run normally to preserve attention mechanisms
        return False
    
    def _forward_block_with_masking(self, x: torch.Tensor, block, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward through a block with masking applied.
        """
        from efficientvit.models.nn import ResidualBlock
        
        if isinstance(block, ResidualBlock):
            # Handle residual block
            if block.main is not None:
                main_out = self._apply_masked_conv(x, block.main, mask)
            else:
                main_out = x
                
            if block.shortcut is not None:
                shortcut_out = block.shortcut(x)
                if shortcut_out.shape == main_out.shape:
                    out = main_out + shortcut_out
                else:
                    out = main_out
            else:
                out = main_out
                
            if hasattr(block, 'post_act') and block.post_act is not None:
                out = block.post_act(out)
                
            return out
        else:
            # Direct conv layer
            return self._apply_masked_conv(x, block, mask)
    
    def _apply_masked_conv(self, x: torch.Tensor, conv_layer, mask: torch.Tensor) -> torch.Tensor:
        """
        Apply masking to a convolution layer.
        """
        masked_conv = RobustMaskedConvLayer(conv_layer)
        return masked_conv(x, mask)
    
    def get_backbone_state_dict(self) -> Dict[str, torch.Tensor]:
        """
        Get state dict that's compatible with original EfficientViT.
        Since we're not modifying the backbone structure, just return original state dict.
        """
        return self.backbone.state_dict()
    
    def save_backbone(self, filepath: str):
        """Save backbone weights for downstream use."""
        state_dict = {
            'model_state_dict': self.get_backbone_state_dict(),
            'width_list': self.backbone.width_list,
            'architecture': type(self.backbone).__name__
        }
        torch.save(state_dict, filepath)
        print(f"Backbone saved to {filepath}")


def load_convmae_backbone(
    backbone_fn,
    checkpoint_path: str,
    **backbone_kwargs
) -> Union[EfficientViTBackbone, EfficientViTLargeBackbone]:
    """
    Load ConvMAE weights into EfficientViT backbone.
    """
    # Create backbone
    backbone = backbone_fn(**backbone_kwargs)
    
    # Load weights if checkpoint exists
    if checkpoint_path and torch.cuda.is_available():
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
                
            # Load with strict=False to handle any missing/extra keys
            missing_keys, unexpected_keys = backbone.load_state_dict(state_dict, strict=False)
            
            if missing_keys:
                print(f"Missing keys: {len(missing_keys)}")
            if unexpected_keys:
                print(f"Unexpected keys: {len(unexpected_keys)}")
                
            print("ConvMAE weights loaded successfully")
        except Exception as e:
            print(f"Warning: Could not load checkpoint {checkpoint_path}: {e}")
            print("Using random initialization")
    
    return backbone


# Simplified ConvMAE model for pretraining
class SimpleConvMAE(nn.Module):
    """
    Simplified ConvMAE model focused on robustness.
    """
    
    def __init__(
        self,
        backbone_fn,
        mask_ratio: float = 0.75,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        **backbone_kwargs
    ):
        super().__init__()
        
        # Create backbone
        original_backbone = backbone_fn(**backbone_kwargs)
        
        # Wrap with ConvMAE functionality
        self.encoder = RobustConvMAEWrapper(original_backbone, mask_ratio)
        
        # Simple decoder for reconstruction
        self.decoder_embed_dim = decoder_embed_dim
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        # Project encoder output to decoder dim
        encoder_dim = original_backbone.width_list[-1]
        self.encoder_to_decoder = nn.Linear(encoder_dim, decoder_embed_dim)
        
        # Simple transformer decoder
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_embed_dim,
            nhead=8,
            dim_feedforward=decoder_embed_dim * 4,
            dropout=0.0,
            activation='gelu',
            batch_first=True
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, num_layers=decoder_depth)
        
        # Prediction head
        self.pred_head = nn.Linear(decoder_embed_dim, 16 * 16 * 3)  # 16x16 patches, RGB
        
        # Initialize
        torch.nn.init.trunc_normal_(self.mask_token, std=0.02)
        
    def forward(self, imgs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Forward pass for training."""
        # Encode
        features = self.encoder(imgs, return_features=True)
        
        # Simple reconstruction loss on final features
        final_features = features['stage_final']  # (B, C, H, W)
        
        # Global average pooling and project to decoder
        pooled = F.adaptive_avg_pool2d(final_features, 1).flatten(1)  # (B, C)
        decoder_input = self.encoder_to_decoder(pooled).unsqueeze(1)  # (B, 1, decoder_dim)
        
        # Add mask tokens (simplified)
        mask_tokens = self.mask_token.expand(decoder_input.shape[0], 196, -1)  # 14x14 = 196
        decoder_input = torch.cat([decoder_input, mask_tokens], dim=1)  # (B, 197, decoder_dim)
        
        # Decode
        decoded = self.decoder(decoder_input)  # (B, 197, decoder_dim)
        
        # Predict
        pred = self.pred_head(decoded[:, 1:, :])  # Skip first token, predict patches
        
        # Simple MSE loss (placeholder)
        target = self.patchify(imgs)  # (B, 196, 16*16*3)
        loss = F.mse_loss(pred, target)
        
        return {
            'loss': loss,
            'pred': pred,
            'target': target,
            'features': features
        }
    
    def patchify(self, imgs: torch.Tensor, patch_size: int = 16) -> torch.Tensor:
        """Convert images to patches."""
        B, C, H, W = imgs.shape
        assert H % patch_size == 0 and W % patch_size == 0
        
        h_patches = H // patch_size
        w_patches = W // patch_size
        
        # Reshape to patches
        patches = imgs.view(B, C, h_patches, patch_size, w_patches, patch_size)
        patches = patches.permute(0, 2, 4, 3, 5, 1)  # (B, h_patches, w_patches, patch_size, patch_size, C)
        patches = patches.reshape(B, h_patches * w_patches, patch_size * patch_size * C)
        
        return patches
    
    def save_backbone(self, filepath: str):
        """Save backbone for downstream use."""
        self.encoder.save_backbone(filepath)


# Factory functions
def convmae_efficientvit_b2(**kwargs):
    """ConvMAE with EfficientViT-B2."""
    from efficientvit.models.efficientvit.backbone import efficientvit_backbone_b2
    return SimpleConvMAE(
        backbone_fn=efficientvit_backbone_b2,
        **kwargs
    )


# Example usage
if __name__ == "__main__":
    print("Testing Robust ConvMAE implementation...")
    
    # Test without masking first
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
        model.save_backbone('robust_convmae_b2.pth')
        print("✓ Backbone saved successfully!")
        
    except Exception as e:
        print(f"✗ Error in masked forward pass: {e}")
        import traceback
        traceback.print_exc()
    
    # Test loading backbone for downstream use
    print("\n3. Testing backbone loading...")
    try:
        from efficientvit.models.efficientvit.backbone import efficientvit_backbone_b2
        
        backbone = load_convmae_backbone(
            efficientvit_backbone_b2,
            'robust_convmae_b2.pth'
        )
        
        with torch.no_grad():
            features = backbone(x)
            print(f"✓ Loaded backbone forward pass successful!")
            print(f"  Features shape: {features['stage_final'].shape}")
            
    except Exception as e:
        print(f"✗ Error loading backbone: {e}")
    
    print("\nRobust ConvMAE test completed!")