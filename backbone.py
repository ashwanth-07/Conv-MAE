"""
Refactored ConvMAE implementation that preserves all original functionality.
Key principle: Don't break what works - only improve organization and clarity.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Dict, Optional, Union

# Import official EfficientViT components
from efficientvit.models.efficientvit.backbone import EfficientViTBackbone, EfficientViTLargeBackbone


class MaskedConvolutionLayer(nn.Module):
    """
    Robust masked convolution that handles dimension mismatches gracefully.
    
    This wrapper applies masking to any convolution layer by:
    1. Ensuring mask dimensions match input
    2. Zeroing out masked regions in input
    3. Applying convolution
    4. Zeroing out corresponding regions in output (conservative approach)
    """
    
    def __init__(self, conv_layer: nn.Module):
        super().__init__()
        self.conv_layer = conv_layer
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with optional masking.
        
        Args:
            x: Input tensor [B, C, H, W]
            mask: Boolean mask [B, 1, H', W'] where True = masked (remove)
            
        Returns:
            Masked convolution output
        """
        if mask is None:
            return self.conv_layer(x)
        
        # Ensure mask matches input spatial dimensions
        if mask.shape[2:] != x.shape[2:]:
            mask = F.interpolate(
                mask.float(), 
                size=x.shape[2:], 
                mode='nearest'
            ).bool()
        
        # Apply mask: zero out masked regions (True = mask out)
        inverse_mask = (~mask).float()
        masked_input = x * inverse_mask
        
        # Apply convolution to masked input
        output = self.conv_layer(masked_input)
        
        # Apply mask to output as well (conservative approach)
        if mask.shape[2:] != output.shape[2:]:
            output_mask = F.interpolate(
                mask.float(), 
                size=output.shape[2:], 
                mode='nearest'
            ).bool()
        else:
            output_mask = mask
            
        masked_output = output * (~output_mask).float()
        
        return masked_output


class MaskGenerator:
    """
    Robust mask generator that maintains 4D tensor format for PyTorch compatibility.
    
    Key design principles:
    1. Always maintain 4D format [B, 1, H, W] for F.interpolate compatibility
    2. Generate base mask once, interpolate to different resolutions as needed
    3. Use conservative masking ratio
    """
    
    def __init__(self, mask_ratio: float = 0.75):
        """
        Initialize mask generator.
        
        Args:
            mask_ratio: Fraction of tokens to mask (0.0 = no masking, 1.0 = mask everything)
        """
        self.mask_ratio = mask_ratio
        
    def generate_base_mask(
        self, 
        batch_size: int, 
        height: int, 
        width: int, 
        device: torch.device
    ) -> torch.Tensor:
        """
        Generate base mask at specified resolution.
        
        Args:
            batch_size: Number of samples in batch
            height: Mask height
            width: Mask width  
            device: Target device
            
        Returns:
            4D boolean mask [B, 1, H, W] where True = masked (remove)
        """
        total_tokens = height * width
        tokens_to_keep = int((1 - self.mask_ratio) * total_tokens)
        
        # Initialize mask - True means masked (remove)
        mask_flat = torch.ones(batch_size, total_tokens, device=device, dtype=torch.bool)
        
        # For each sample in batch, randomly select tokens to keep
        for batch_idx in range(batch_size):
            keep_indices = torch.randperm(total_tokens, device=device)[:tokens_to_keep]
            mask_flat[batch_idx, keep_indices] = False  # False = keep, True = mask
            
        # Reshape to 4D format [B, 1, H, W] - CRITICAL for F.interpolate compatibility
        base_mask = mask_flat.view(batch_size, 1, height, width)
        
        return base_mask
    
    def get_mask_for_resolution(
        self, 
        base_mask: torch.Tensor, 
        target_height: int, 
        target_width: int
    ) -> torch.Tensor:
        """
        Interpolate base mask to target resolution.
        
        Args:
            base_mask: 4D base mask [B, 1, H, W]
            target_height: Target height
            target_width: Target width
            
        Returns:
            4D mask at target resolution [B, 1, target_H, target_W]
        """
        current_height, current_width = base_mask.shape[2], base_mask.shape[3]
        
        # If already correct size, return as-is
        if current_height == target_height and current_width == target_width:
            return base_mask
        
        # Interpolate to target size - works because both input and output are 4D/2D respectively
        resized_mask = F.interpolate(
            base_mask.float(), 
            size=(target_height, target_width), 
            mode='nearest'
        ).bool()
        
        return resized_mask


class ConvMAEWrapper(nn.Module):
    """
    Conservative ConvMAE wrapper that applies masking selectively.
    
    Design philosophy:
    - Only mask layers that are safe (conv layers)
    - Preserve attention mechanisms unchanged  
    - Use base mask approach for efficiency
    - Maintain full compatibility with original backbone
    """
    
    def __init__(
        self, 
        backbone: Union[EfficientViTBackbone, EfficientViTLargeBackbone], 
        mask_ratio: float = 0.75
    ):
        """
        Initialize ConvMAE wrapper.
        
        Args:
            backbone: EfficientViT backbone to wrap
            mask_ratio: Masking ratio for training
        """
        super().__init__()
        
        self.backbone = backbone
        self.mask_ratio = mask_ratio
        self.mask_generator = MaskGenerator(mask_ratio)
        
        # State variables
        self.base_mask = None
        self.training = True
        
    def __call__(
        self, 
        x: torch.Tensor, 
        return_features: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Make wrapper callable like original backbone."""
        return self.forward(x, return_features)
        
    def train(self, mode: bool = True):
        """Set training mode for both wrapper and backbone."""
        self.training = mode
        self.backbone.train(mode)
        return self
        
    def eval(self):
        """Set evaluation mode for both wrapper and backbone."""
        self.training = False
        self.backbone.eval()
        return self
        
    def forward(
        self, 
        x: torch.Tensor, 
        return_features: bool = False
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass with selective masking.
        
        Args:
            x: Input tensor [B, C, H, W]
            return_features: Whether to return intermediate features
            
        Returns:
            Either final features or dict of all features
        """
        batch_size, channels, height, width = x.shape
        
        # Generate base mask only during training and if masking enabled
        if self.training and self.mask_ratio > 0:
            # Use conservative base resolution (final stage resolution)
            # Assuming 16x total downsampling for typical EfficientViT
            base_height, base_width = height // 16, width // 16
            self.base_mask = self.mask_generator.generate_base_mask(
                batch_size, base_height, base_width, x.device
            )
        else:
            self.base_mask = None
            
        # Forward through backbone with selective masking
        features = self._forward_with_selective_masking(x)
        
        # Return format depends on request
        if return_features:
            if self.base_mask is not None:
                features['base_mask'] = self.base_mask
            return features
        else:
            return features['stage_final']
    
    def _forward_with_selective_masking(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass with selective masking applied only to safe layers.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary of features from each stage
        """
        features = {"input": x}
        current_tensor = x
        
        # Forward through input stem with optional masking
        current_tensor = self._forward_stage_with_masking(
            current_tensor, 
            self.backbone.input_stem, 
            "input_stem"
        )
        features["stage0"] = current_tensor
        
        # Forward through all stages
        for stage_index, stage in enumerate(self.backbone.stages, 1):
            current_tensor = self._forward_stage_with_masking(
                current_tensor, 
                stage, 
                f"stage{stage_index}"
            )
            features[f"stage{stage_index}"] = current_tensor
            
        features["stage_final"] = current_tensor
        return features
    
    def _forward_stage_with_masking(
        self, 
        x: torch.Tensor, 
        stage: nn.Module, 
        stage_name: str
    ) -> torch.Tensor:
        """
        Forward through a stage with optional masking.
        
        Only applies masking to conv layers, leaves attention layers unchanged.
        
        Args:
            x: Input tensor
            stage: Stage module to forward through
            stage_name: Name of stage (for debugging)
            
        Returns:
            Output tensor after stage processing
        """
        # If no masking, use standard forward pass
        if self.base_mask is None:
            return stage(x)
        
        current_tensor = x
        
        # Process each block in the stage
        for block in stage.op_list:
            if self._is_safe_to_mask_block(block):
                # Apply masking to this block
                current_mask = self.mask_generator.get_mask_for_resolution(
                    self.base_mask, 
                    current_tensor.shape[2], 
                    current_tensor.shape[3]
                )
                current_tensor = self._forward_block_with_masking(
                    current_tensor, 
                    block, 
                    current_mask
                )
            else:
                # Forward without masking (e.g., attention blocks)
                current_tensor = block(current_tensor)
                
        return current_tensor
    
    def _is_safe_to_mask_block(self, block: nn.Module) -> bool:
        """
        Determine if a block is safe to apply masking to.
        
        Conservative approach: only mask pure convolution layers.
        Attention mechanisms are left unchanged to preserve their function.
        
        Args:
            block: Module to check
            
        Returns:
            True if safe to mask, False otherwise
        """
        # Import here to avoid circular imports
        from efficientvit.models.nn import ConvLayer, MBConv, DSConv, ResidualBlock
        
        # Safe to mask: pure convolution layers
        if isinstance(block, (ConvLayer, MBConv, DSConv)):
            return True
            
        # Safe to mask: residual blocks with conv main blocks
        if isinstance(block, ResidualBlock):
            if hasattr(block, 'main') and isinstance(block.main, (ConvLayer, MBConv, DSConv)):
                return True
                
        # NOT safe to mask: EfficientViTBlock (hybrid attention/conv)
        # These contain attention mechanisms that should run normally
        return False
    
    def _forward_block_with_masking(
        self, 
        x: torch.Tensor, 
        block: nn.Module, 
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward through a block with masking applied.
        
        Args:
            x: Input tensor
            block: Block to forward through  
            mask: Mask to apply
            
        Returns:
            Output tensor after masked block processing
        """
        from efficientvit.models.nn import ResidualBlock
        
        if isinstance(block, ResidualBlock):
            # Handle residual block carefully
            main_output = x
            if hasattr(block, 'main') and block.main is not None:
                main_output = self._apply_masked_convolution(x, block.main, mask)
                
            # Handle shortcut connection
            shortcut_output = x
            if hasattr(block, 'shortcut') and block.shortcut is not None:
                shortcut_output = block.shortcut(x)
                
            # Combine main and shortcut if dimensions match
            if shortcut_output.shape == main_output.shape:
                combined_output = main_output + shortcut_output
            else:
                combined_output = main_output
                
            # Apply post-activation if exists
            if hasattr(block, 'post_act') and block.post_act is not None:
                combined_output = block.post_act(combined_output)
                
            return combined_output
        else:
            # Direct convolution layer
            return self._apply_masked_convolution(x, block, mask)
    
    def _apply_masked_convolution(
        self, 
        x: torch.Tensor, 
        conv_layer: nn.Module, 
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Apply masking to a convolution layer.
        
        Args:
            x: Input tensor
            conv_layer: Convolution layer
            mask: Mask to apply
            
        Returns:
            Masked convolution output
        """
        masked_conv_layer = MaskedConvolutionLayer(conv_layer)
        return masked_conv_layer(x, mask)
    
    def get_backbone_state_dict(self) -> Dict[str, torch.Tensor]:
        """
        Get state dict that's compatible with original EfficientViT.
        
        Since we don't modify the backbone structure, return original state dict.
        
        Returns:
            Original backbone state dict
        """
        return self.backbone.state_dict()
    
    def save_backbone(self, filepath: str):
        """
        Save backbone weights for downstream use.
        
        Args:
            filepath: Path to save backbone weights
        """
        state_dict = {
            'model_state_dict': self.get_backbone_state_dict(),
            'width_list': self.backbone.width_list,
            'architecture': type(self.backbone).__name__
        }
        torch.save(state_dict, filepath)
        print(f"Backbone saved to {filepath}")


def load_convmae_backbone(
    backbone_function,
    checkpoint_path: str,
    **backbone_kwargs
) -> Union[EfficientViTBackbone, EfficientViTLargeBackbone]:
    """
    Load ConvMAE weights into EfficientViT backbone.
    
    Args:
        backbone_function: Function to create backbone
        checkpoint_path: Path to checkpoint file
        **backbone_kwargs: Additional arguments for backbone creation
        
    Returns:
        Loaded backbone
    """
    # Create fresh backbone
    backbone = backbone_function(**backbone_kwargs)
    
    # Load weights if checkpoint exists
    if checkpoint_path and torch.cuda.is_available():
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Extract state dict from checkpoint
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
                
            # Load with flexible key matching
            missing_keys, unexpected_keys = backbone.load_state_dict(
                state_dict, 
                strict=False
            )
            
            # Report loading status
            if missing_keys:
                print(f"Missing keys: {len(missing_keys)}")
            if unexpected_keys:
                print(f"Unexpected keys: {len(unexpected_keys)}")
                
            print("ConvMAE weights loaded successfully")
            
        except Exception as e:
            print(f"Warning: Could not load checkpoint {checkpoint_path}: {e}")
            print("Using random initialization")
    
    return backbone


class SimpleConvMAE(nn.Module):
    """
    Simplified ConvMAE model for pretraining.
    
    Combines ConvMAE encoder with simple transformer decoder for reconstruction.
    """
    
    def __init__(
        self,
        backbone_function,
        mask_ratio: float = 0.75,
        decoder_embed_dim: int = 512,
        decoder_depth: int = 8,
        **backbone_kwargs
    ):
        """
        Initialize SimpleConvMAE.
        
        Args:
            backbone_function: Function to create backbone
            mask_ratio: Masking ratio for encoder
            decoder_embed_dim: Decoder embedding dimension
            decoder_depth: Number of decoder layers
            **backbone_kwargs: Arguments for backbone creation
        """
        super().__init__()
        
        # Create backbone and wrap with ConvMAE
        original_backbone = backbone_function(**backbone_kwargs)
        self.encoder = ConvMAEWrapper(original_backbone, mask_ratio)
        
        # Decoder configuration
        self.decoder_embed_dim = decoder_embed_dim
        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
        
        # Project encoder output to decoder dimension
        encoder_output_dim = original_backbone.width_list[-1]
        self.encoder_to_decoder = nn.Linear(encoder_output_dim, decoder_embed_dim)
        
        # Simple transformer decoder
        decoder_layer = nn.TransformerEncoderLayer(
            d_model=decoder_embed_dim,
            nhead=8,
            dim_feedforward=decoder_embed_dim * 4,
            dropout=0.0,
            activation='gelu',
            batch_first=True
        )
        self.decoder = nn.TransformerEncoder(
            decoder_layer, 
            num_layers=decoder_depth
        )
        
        # Prediction head for patch reconstruction
        patch_size = 16
        self.prediction_head = nn.Linear(
            decoder_embed_dim, 
            patch_size * patch_size * 3  # RGB patches
        )
        
        # Initialize parameters
        torch.nn.init.trunc_normal_(self.mask_token, std=0.02)
        
    def forward(self, images: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass for training.
        
        Args:
            images: Input images [B, 3, H, W]
            
        Returns:
            Dictionary with loss, predictions, targets, and features
        """
        # Encode with masking
        features = self.encoder(images, return_features=True)
        
        # Use final stage features for reconstruction
        final_features = features['stage_final']  # [B, C, H, W]
        
        # Global average pooling and project to decoder dimension
        pooled_features = F.adaptive_avg_pool2d(final_features, 1).flatten(1)  # [B, C]
        decoder_input = self.encoder_to_decoder(pooled_features).unsqueeze(1)  # [B, 1, decoder_dim]
        
        # Add mask tokens (simplified approach)
        num_patches = 196  # 14x14 patches for 224x224 images with 16x16 patches
        mask_tokens = self.mask_token.expand(
            decoder_input.shape[0], 
            num_patches, 
            -1
        )
        decoder_input = torch.cat([decoder_input, mask_tokens], dim=1)  # [B, 197, decoder_dim]
        
        # Decode
        decoded_features = self.decoder(decoder_input)  # [B, 197, decoder_dim]
        
        # Predict patches (skip first token which is encoded feature)
        patch_predictions = self.prediction_head(decoded_features[:, 1:, :])  # [B, 196, patch_dim]
        
        # Create target patches
        target_patches = self._convert_images_to_patches(images)  # [B, 196, patch_dim]
        
        # Compute reconstruction loss
        reconstruction_loss = F.mse_loss(patch_predictions, target_patches)
        
        return {
            'loss': reconstruction_loss,
            'predictions': patch_predictions,
            'targets': target_patches,
            'features': features
        }
    
    def _convert_images_to_patches(
        self, 
        images: torch.Tensor, 
        patch_size: int = 16
    ) -> torch.Tensor:
        """
        Convert images to patches for reconstruction target.
        
        Args:
            images: Input images [B, C, H, W]
            patch_size: Size of each patch
            
        Returns:
            Patches [B, num_patches, patch_size*patch_size*C]
        """
        batch_size, channels, height, width = images.shape
        assert height % patch_size == 0 and width % patch_size == 0, \
            f"Image dimensions ({height}, {width}) must be divisible by patch_size ({patch_size})"
        
        patches_per_height = height // patch_size
        patches_per_width = width // patch_size
        
        # Reshape to patches
        patches = images.view(
            batch_size, 
            channels, 
            patches_per_height, 
            patch_size, 
            patches_per_width, 
            patch_size
        )
        
        # Rearrange to [B, patches_per_height, patches_per_width, patch_size, patch_size, C]
        patches = patches.permute(0, 2, 4, 3, 5, 1)
        
        # Flatten to [B, num_patches, patch_dim]
        num_patches = patches_per_height * patches_per_width
        patch_dim = patch_size * patch_size * channels
        patches = patches.reshape(batch_size, num_patches, patch_dim)
        
        return patches
    
    def save_backbone(self, filepath: str):
        """Save backbone for downstream use."""
        self.encoder.save_backbone(filepath)


# Factory functions for easy model creation
def convmae_efficientvit_b2(**kwargs):
    """Create ConvMAE with EfficientViT-B2 backbone."""
    from efficientvit.models.efficientvit.backbone import efficientvit_backbone_b2
    return SimpleConvMAE(
        backbone_function=efficientvit_backbone_b2,
        **kwargs
    )


# Example usage and testing
if __name__ == "__main__":
    print("Testing Refactored ConvMAE implementation...")
    
    # Test 1: Without masking
    print("\n1. Testing without masking...")
    model = convmae_efficientvit_b2(mask_ratio=0.0)
    model.eval()
    
    test_input = torch.randn(2, 3, 224, 224)
    
    with torch.no_grad():
        output = model(test_input)
        print(f"✓ Forward pass successful! Loss: {output['loss'].item():.4f}")
    
    # Test 2: With masking
    print("\n2. Testing with masking...")
    model = convmae_efficientvit_b2(mask_ratio=0.75)
    model.train()
    
    try:
        output = model(test_input)
        print(f"✓ Masked forward pass successful! Loss: {output['loss'].item():.4f}")
        
        # Save backbone
        model.save_backbone('refactored_convmae_b2.pth')
        print("✓ Backbone saved successfully!")
        
    except Exception as e:
        print(f"✗ Error in masked forward pass: {e}")
        import traceback
        traceback.print_exc()
    
    # Test 3: Backbone loading
    print("\n3. Testing backbone loading...")
    try:
        from efficientvit.models.efficientvit.backbone import efficientvit_backbone_b2
        
        backbone = load_convmae_backbone(
            efficientvit_backbone_b2,
            'refactored_convmae_b2.pth'
        )
        
        with torch.no_grad():
            features = backbone(test_input)
            print(f"✓ Loaded backbone forward pass successful!")
            print(f"  Features shape: {features['stage_final'].shape}")
            
    except Exception as e:
        print(f"✗ Error loading backbone: {e}")
    
    print("\nRefactored ConvMAE test completed!")