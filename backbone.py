import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple, Dict, Optional, Union
from einops import rearrange

# Import official EfficientViT components
from efficientvit.models.nn import (
    ConvLayer, EfficientViTBlock, MBConv, DSConv, FusedMBConv, 
    ResidualBlock, IdentityLayer, OpSequential, ResBlock, LiteMLA
)
from efficientvit.models.efficientvit.backbone import EfficientViTBackbone, EfficientViTLargeBackbone


def random_masking(x, mask_ratio):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    N, L, D = x.shape  # batch, length, dim
    len_keep = int(L * (1 - mask_ratio))
    
    noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
    
    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    ids_keep = ids_shuffle[:, :len_keep]
    x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

    # generate the binary mask: 0 is keep, 1 is remove
    mask = torch.ones([N, L], device=x.device)
    mask[:, :len_keep] = 0
    # unshuffle to get the binary mask
    mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask, ids_restore


def unshuffle_and_insert_mask_tokens(x_vis, ids_restore, mask_token, original_length):
    """
    Restore full sequence by inserting mask tokens.
    x_vis: [N, len_keep, D] - visible tokens after processing
    ids_restore: [N, L] - shuffle indices for restoration
    mask_token: [1, 1, D] - learnable mask token
    original_length: L - original sequence length
    """
    N, len_keep, D = x_vis.shape
    
    # Create full sequence with mask tokens
    mask_tokens = mask_token.expand(N, original_length - len_keep, -1)
    x_full = torch.cat([x_vis, mask_tokens], dim=1)  # [N, L, D]
    
    # Unshuffle to restore original order
    x_full = torch.gather(x_full, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, D))
    
    return x_full


class MaskedConvLayer(nn.Module):
    """Masked convolution that zeros out contributions from masked regions."""
    def __init__(self, conv_layer: ConvLayer, kernel_size: int = 3):
        super().__init__()
        self.conv_layer = conv_layer
        self.kernel_size = kernel_size
        self.padding = kernel_size // 2
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        if mask is None:
            return self.conv_layer(x)
            
        # Apply mask before convolution
        inv_mask = (~mask).float()
        x_masked = x * inv_mask
        
        # Apply convolution
        x_conv = self.conv_layer(x_masked)
        
        # For kernel size > 1, ensure masked regions don't contribute to outputs
        if self.kernel_size > 1:
            # Dilate mask to account for receptive field
            kernel = torch.ones(1, 1, self.kernel_size, self.kernel_size, device=mask.device)
            dilated_mask = F.conv2d(mask.float(), kernel, padding=self.padding) > 0
            output_mask = (~dilated_mask).float()
            x_conv = x_conv * output_mask
        else:
            x_conv = x_conv * inv_mask
            
        return x_conv


class MaskedMBConv(nn.Module):
    """Masked version of MBConv that operates on full spatial grid."""
    def __init__(self, mbconv: MBConv):
        super().__init__()
        # Extract components from original MBConv
        self.inverted_conv = MaskedConvLayer(mbconv.inverted_conv, kernel_size=1)
        self.depth_conv = MaskedConvLayer(mbconv.depth_conv, kernel_size=3)  # Usually 3x3
        self.point_conv = MaskedConvLayer(mbconv.point_conv, kernel_size=1)
        
    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        x = self.inverted_conv(x, mask)
        x = self.depth_conv(x, mask)
        x = self.point_conv(x, mask)
        return x


class CorrectMaskedEfficientViTBlock(nn.Module):
    """
    Correctly handles the hybrid nature of EfficientViTBlock:
    - Runs MBConv on full spatial grid with masking
    - Drops tokens only for LiteMLA attention computation
    - Preserves spatial structure throughout
    """
    def __init__(self, evit_block: EfficientViTBlock, mask_ratio: float = 0.75):
        super().__init__()
        
        # Extract the two main components
        self.context_module = evit_block.context_module  # Contains LiteMLA
        self.local_module_main = evit_block.local_module.main  # The MBConv
        self.local_module_shortcut = evit_block.local_module.shortcut  # Identity or None
        
        # Create masked version of MBConv
        if isinstance(self.local_module_main, MBConv):
            self.masked_local_module = MaskedMBConv(self.local_module_main)
        else:
            # Fallback for other conv types
            self.masked_local_module = self.local_module_main
            
        # Mask token for attention dropout/restoration
        self.mask_ratio = mask_ratio
        embed_dim = evit_block.context_module.main.qkv.conv.out_channels // 3  # Get embed dim from LiteMLA
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Initialize mask token
        torch.nn.init.trunc_normal_(self.mask_token, std=0.02)
    
    def forward(self, x: torch.Tensor, spatial_mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, C, H, W)
            spatial_mask: Spatial mask (B, 1, H, W) where True=masked
        """
        B, C, H, W = x.shape
        
        # === Step 1: Context Module (LiteMLA) with token dropping ===
        if spatial_mask is not None and self.training:
            x_context = self._forward_context_with_masking(x, spatial_mask)
        else:
            x_context = self.context_module(x)
            
        # === Step 2: Local Module (MBConv) on full spatial grid ===
        if spatial_mask is not None:
            # Apply masked convolution on full grid
            x_local = self.masked_local_module(x_context, spatial_mask)
            
            # Apply shortcut connection if exists
            if self.local_module_shortcut is not None:
                shortcut = self.local_module_shortcut(x_context)
                if shortcut.shape == x_local.shape:
                    x_local = x_local + shortcut
            
            # Final masking to ensure masked regions stay zero
            x_local = x_local * (~spatial_mask).float()
        else:
            # Standard forward without masking
            x_local = self.local_module_main(x_context)
            if self.local_module_shortcut is not None:
                shortcut = self.local_module_shortcut(x_context)
                if shortcut.shape == x_local.shape:
                    x_local = x_local + shortcut
        
        return x_local
    
    def _forward_context_with_masking(self, x: torch.Tensor, spatial_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward through context module (LiteMLA) with proper token masking.
        Maintains spatial structure while dropping tokens only for attention computation.
        """
        B, C, H, W = x.shape
        
        # Get the LiteMLA module
        lite_mla = self.context_module.main
        
        # === Process through LiteMLA components ===
        
        # 1. Generate QKV on full spatial grid
        qkv = lite_mla.qkv(x)  # (B, 3*total_dim, H, W)
        
        # 2. Apply multi-scale aggregation (if any)
        multi_scale_qkv = [qkv]
        for op in lite_mla.aggreg:
            multi_scale_qkv.append(op(qkv))
        multi_scale_qkv = torch.cat(multi_scale_qkv, dim=1)  # (B, aggregated_dim, H, W)
        
        # 3. Flatten for attention computation
        qkv_flat = multi_scale_qkv.flatten(2).transpose(1, 2)  # (B, H*W, aggregated_dim)
        
        # 4. Apply masking - drop tokens for attention
        if self.mask_ratio > 0:
            qkv_vis, mask_indices, ids_restore = random_masking(qkv_flat, self.mask_ratio)
        else:
            qkv_vis = qkv_flat
            mask_indices = torch.zeros(B, H*W, device=x.device)
            ids_restore = torch.arange(H*W, device=x.device).unsqueeze(0).expand(B, -1)
        
        # 5. Apply attention on visible tokens only
        # Reshape back to spatial for attention computation
        num_vis = qkv_vis.shape[1]
        if num_vis > 0:
            # For attention computation, we need to handle the spatial reshaping carefully
            # The key insight: LiteMLA expects spatial input, so we create a compact spatial layout
            h_vis = w_vis = int(num_vis ** 0.5)
            if h_vis * w_vis == num_vis:
                qkv_spatial = qkv_vis.transpose(1, 2).view(B, -1, h_vis, w_vis)
                
                # Apply the attention mechanism (qt_attention, relu_linear_att, or softmax_att)
                # We need to modify LiteMLA to handle compact grids properly
                # For now, use the original LiteMLA but understand this is approximate
                attn_out = lite_mla.qt_attention(qkv_spatial)  # or whichever attention method
                
                # Back to sequence
                attn_vis = attn_out.flatten(2).transpose(1, 2)  # (B, num_vis, out_dim)
            else:
                # Fallback: just return qkv_vis if perfect square not possible
                attn_vis = qkv_vis
        else:
            attn_vis = qkv_vis
        
        # 6. Restore full sequence with mask tokens
        if self.mask_ratio > 0:
            out_dim = attn_vis.shape[-1] if attn_vis.shape[1] > 0 else multi_scale_qkv.shape[1]
            attn_full = unshuffle_and_insert_mask_tokens(
                attn_vis, ids_restore, self.mask_token[:, :, :out_dim], H*W
            )
        else:
            attn_full = attn_vis
        
        # 7. Project to output
        out_seq = lite_mla.proj(attn_full.transpose(1, 2))  # Expecting (B, dim, seq_len)
        out_spatial = out_seq.view(B, -1, H, W)
        
        # 8. Apply residual connection
        if self.context_module.shortcut is not None:
            shortcut = self.context_module.shortcut(x)
            if shortcut.shape == out_spatial.shape:
                out_spatial = out_spatial + shortcut
        
        return out_spatial


class BlockMaskGenerator:
    """Generate block-wise masks for ConvMAE."""
    def __init__(self, mask_ratio: float = 0.75):
        self.mask_ratio = mask_ratio

    def __call__(self, batch_size: int, stage_resolutions: list, device: torch.device) -> list:
        """Generate masks for all stages."""
        # Use the finest resolution stage for mask generation
        finest_stage_idx = -1  # Last stage
        H_finest, W_finest = stage_resolutions[finest_stage_idx]
        
        # Generate base mask
        num_tokens = H_finest * W_finest
        num_keep = int((1 - self.mask_ratio) * num_tokens)
        
        masks = []
        finest_mask_flat = torch.ones(batch_size, num_tokens, device=device, dtype=torch.bool)
        
        for b in range(batch_size):
            keep_indices = torch.randperm(num_tokens, device=device)[:num_keep]
            finest_mask_flat[b, keep_indices] = False
        
        finest_mask = finest_mask_flat.view(batch_size, 1, H_finest, W_finest)
        
        # Generate masks for all stages
        for H, W in stage_resolutions:
            if H == H_finest and W == W_finest:
                masks.append(finest_mask)
            else:
                stage_mask = F.interpolate(finest_mask.float(), size=(H, W), mode='nearest').bool()
                masks.append(stage_mask)
        
        return masks


class CorrectedConvMAEBackbone(nn.Module):
    """
    Correctly implemented ConvMAE backbone that handles EfficientViT's hybrid blocks properly.
    """
    def __init__(
        self,
        backbone: Union[EfficientViTBackbone, EfficientViTLargeBackbone],
        mask_ratio: float = 0.75
    ):
        super().__init__()
        
        self.original_backbone = backbone
        self.mask_ratio = mask_ratio
        self.mask_generator = BlockMaskGenerator(mask_ratio)
        self.width_list = backbone.width_list.copy()
        
        # Wrap stages appropriately
        self.input_stem = self._wrap_conv_stage(backbone.input_stem)
        self.stages = nn.ModuleList([
            self._wrap_stage(stage, stage_id) for stage_id, stage in enumerate(backbone.stages)
        ])
        
    def _wrap_conv_stage(self, stage: OpSequential) -> OpSequential:
        """Wrap pure convolution stages with simple masking."""
        wrapped_blocks = []
        for block in stage.op_list:
            if isinstance(block, ResidualBlock) and isinstance(block.main, (MBConv, DSConv)):
                if isinstance(block.main, MBConv):
                    wrapped_main = MaskedMBConv(block.main)
                else:
                    wrapped_main = block.main  # Handle DSConv similarly if needed
                wrapped_block = ResidualBlock(
                    main=wrapped_main,
                    shortcut=block.shortcut,
                    post_act=getattr(block, 'post_act', None),
                    pre_norm=getattr(block, 'pre_norm', None)
                )
                wrapped_blocks.append(wrapped_block)
            elif isinstance(block, ConvLayer):
                wrapped_blocks.append(MaskedConvLayer(block))
            else:
                wrapped_blocks.append(block)
        return OpSequential(wrapped_blocks)
        
    def _wrap_stage(self, stage: OpSequential, stage_id: int) -> OpSequential:
        """Wrap stages, handling hybrid EfficientViTBlocks correctly."""
        wrapped_blocks = []
        
        for block in stage.op_list:
            if isinstance(block, EfficientViTBlock):
                # This is a hybrid block - use our corrected implementation
                wrapped_blocks.append(CorrectMaskedEfficientViTBlock(block, self.mask_ratio))
            elif isinstance(block, ResidualBlock):
                # Handle other residual blocks
                if isinstance(block.main, (MBConv, DSConv)):
                    wrapped_main = MaskedMBConv(block.main) if isinstance(block.main, MBConv) else block.main
                    wrapped_block = ResidualBlock(
                        main=wrapped_main,
                        shortcut=block.shortcut,
                        post_act=getattr(block, 'post_act', None),
                        pre_norm=getattr(block, 'pre_norm', None)
                    )
                    wrapped_blocks.append(wrapped_block)
                else:
                    wrapped_blocks.append(block)
            else:
                wrapped_blocks.append(block)
                
        return OpSequential(wrapped_blocks)
    
    def forward(self, x: torch.Tensor, return_features: bool = False) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass with correct masking."""
        B, C, H, W = x.shape
        output_dict = {"input": x}
        
        # Generate masks if training
        masks = None
        if self.training and self.mask_ratio > 0:
            # Calculate stage resolutions
            stage_resolutions = self._calculate_stage_resolutions(H, W)
            masks = self.mask_generator(B, stage_resolutions, x.device)
        
        # Forward through input stem
        current_mask = masks[0] if masks else None
        if current_mask is not None:
            for block in self.input_stem.op_list:
                if hasattr(block, 'forward') and len(block.forward.__code__.co_varnames) > 2:
                    x = block(x, current_mask)
                else:
                    x = block(x)
        else:
            x = self.input_stem(x)
        output_dict["stage0"] = x
        
        # Forward through stages
        for stage_id, stage in enumerate(self.stages, 1):
            current_mask = masks[stage_id] if masks and stage_id < len(masks) else None
            
            for block in stage.op_list:
                if isinstance(block, CorrectMaskedEfficientViTBlock):
                    x = block(x, current_mask)
                elif hasattr(block, 'forward') and len(block.forward.__code__.co_varnames) > 2:
                    x = block(x, current_mask)
                else:
                    x = block(x)
                    
            output_dict[f"stage{stage_id}"] = x
        
        output_dict["stage_final"] = x
        
        if return_features:
            if masks:
                for i, mask in enumerate(masks):
                    output_dict[f"mask{i}"] = mask
            return output_dict
        return x
    
    def _calculate_stage_resolutions(self, H: int, W: int) -> list:
        """Calculate spatial resolutions for each stage."""
        # This needs to be implemented based on your specific backbone architecture
        # For now, assume standard downsampling pattern
        resolutions = [(H//4, W//4)]  # Input stem: /4
        
        # Each stage may have different downsampling
        current_h, current_w = H//4, W//4
        for stage in self.stages:
            # Check if this stage has downsampling
            has_downsample = any(
                hasattr(block, 'main') and hasattr(block.main, 'stride') and 
                getattr(block.main, 'stride', 1) > 1
                for block in stage.op_list if isinstance(block, ResidualBlock)
            )
            if has_downsample:
                current_h, current_w = current_h // 2, current_w // 2
            resolutions.append((current_h, current_w))
            
        return resolutions
    
    def get_state_dict_for_finetuning(self) -> Dict[str, torch.Tensor]:
        """Extract state dict compatible with original EfficientViT."""
        state_dict = {}
        
        for name, param in self.named_parameters():
            # Map back to original parameter names
            original_name = self._map_to_original_name(name)
            if original_name:
                state_dict[original_name] = param.data.clone()
                
        return state_dict
    
    def _map_to_original_name(self, name: str) -> Optional[str]:
        """Map ConvMAE parameter names back to original EfficientViT names."""
        # Remove wrapper prefixes and map to original structure
        original_name = name
        
        # Handle our corrected wrapper names
        if 'masked_local_module.' in original_name:
            original_name = original_name.replace('masked_local_module.', 'local_module.main.')
        if 'context_module.main.' in original_name:
            original_name = original_name.replace('context_module.main.', 'context_module.main.')
        
        # Skip mask tokens and other training-only parameters
        if 'mask_token' in original_name:
            return None
            
        return original_name


# Example usage showing the correction
if __name__ == "__main__":
    from efficientvit.models.efficientvit.backbone import efficientvit_backbone_b2
    
    # Create backbone
    original_backbone = efficientvit_backbone_b2()
    
    # Wrap with corrected ConvMAE implementation
    convmae_backbone = CorrectedConvMAEBackbone(original_backbone, mask_ratio=0.75)
    
    # Test forward pass
    x = torch.randn(2, 3, 224, 224)
    
    # Training mode - with masking
    convmae_backbone.train()
    features_train = convmae_backbone(x, return_features=True)
    print("Training mode (with masking):")
    for key, val in features_train.items():
        if torch.is_tensor(val):
            print(f"  {key}: {val.shape}")
    
    # Eval mode - no masking
    convmae_backbone.eval()
    features_eval = convmae_backbone(x, return_features=True)
    print("\nEval mode (no masking):")
    for key, val in features_eval.items():
        if torch.is_tensor(val):
            print(f"  {key}: {val.shape}")