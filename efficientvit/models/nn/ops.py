# Modifications Copyright © 2025 Advanced Micro Devices, Inc. All rights reserved.
#

import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from einops import rearrange

from efficientvit.models.nn.act import build_act
from efficientvit.models.nn.norm import build_norm
from efficientvit.models.utils import get_same_padding, list_sum, resize, val2list, val2tuple

__all__ = [
    "ConvLayer",
    "UpSampleLayer",
    "LinearLayer",
    "IdentityLayer",
    "DSConv",
    "MBConv",
    "FusedMBConv",
    "ResBlock",
    "LiteMLA",
    "EfficientViTBlock",
    "ResidualBlock",
    "DAGBlock",
    "OpSequential",
]


#################################################################################
#                             Basic Layers                                      #
#################################################################################


class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        use_bias=False,
        dropout=0,
        norm="bn2d",
        act_func="relu",
    ):
        super(ConvLayer, self).__init__()

        padding = get_same_padding(kernel_size)
        padding *= dilation

        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=padding,
            dilation=(dilation, dilation),
            groups=groups,
            bias=use_bias,
        )
        self.norm = build_norm(norm, num_features=out_channels)
        self.act = build_act(act_func)

    def forward(self, x: torch.Tensor, valid_mask = None) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)

        if valid_mask is not None:
            if valid_mask.shape[2:] != x.shape[2:]:
                mask = F.interpolate(valid_mask, size=x.shape[2:], mode='nearest')
            else:
                mask = valid_mask

            x = x * mask
            x = self.conv(x)

            if mask.shape[2:] != x.shape[2:]:
                mask = F.interpolate(mask, size=x.shape[2:], mode='nearest')
            x = x * mask
        else:
            x = self.conv(x)

        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x

class UpSampleLayer(nn.Module):
    def __init__(
        self,
        mode="bicubic",
        size: int or tuple[int, int] or list[int] or None = None,
        factor=2,
        align_corners=False,
    ):
        super(UpSampleLayer, self).__init__()
        self.mode = mode
        self.size = val2list(size, 2) if size is not None else None
        self.factor = None if self.size is not None else factor
        self.align_corners = align_corners

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if (self.size is not None and tuple(x.shape[-2:]) == self.size) or self.factor == 1:
            return x
        return resize(x, self.size, self.factor, self.mode, self.align_corners)


class LinearLayer(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        use_bias=True,
        dropout=0,
        norm=None,
        act_func=None,
    ):
        super(LinearLayer, self).__init__()

        self.dropout = nn.Dropout(dropout, inplace=False) if dropout > 0 else None
        self.linear = nn.Linear(in_features, out_features, use_bias)
        self.norm = build_norm(norm, num_features=out_features)
        self.act = build_act(act_func)

    def _try_squeeze(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() > 2:
            x = torch.flatten(x, start_dim=1)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._try_squeeze(x)
        if self.dropout:
            x = self.dropout(x)
        x = self.linear(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x


class IdentityLayer(nn.Module):
    def forward(self, x: torch.Tensor, valid_mask=None) -> torch.Tensor:
        if valid_mask is not None:
            if valid_mask.shape[2:] != x.shape[2:]:
                mask = F.interpolate(valid_mask, size=x.shape[2:], mode='nearest')
            else:
                mask = valid_mask
            return x * mask
        else:
            return x


#################################################################################
#                             Basic Blocks                                      #
#################################################################################


class DSConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        use_bias=False,
        norm=("bn2d", "bn2d"),
        act_func=("relu6", None),
    ):
        super(DSConv, self).__init__()

        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        self.depth_conv = ConvLayer(
            in_channels,
            in_channels,
            kernel_size,
            stride,
            groups=in_channels,
            norm=norm[0],
            act_func=act_func[0],
            use_bias=use_bias[0],
        )
        self.point_conv = ConvLayer(
            in_channels,
            out_channels,
            1,
            norm=norm[1],
            act_func=act_func[1],
            use_bias=use_bias[1],
        )

    def forward(self, x: torch.Tensor, valid_mask=None) -> torch.Tensor:
        x = self.depth_conv(x, valid_mask)
        x = self.point_conv(x, valid_mask)
        return x


class MBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=6,
        use_bias=False,
        norm=("bn2d", "bn2d", "bn2d"),
        act_func=("relu6", "relu6", None),
    ):
        super(MBConv, self).__init__()

        use_bias = val2tuple(use_bias, 3)
        norm = val2tuple(norm, 3)
        act_func = val2tuple(act_func, 3)
        mid_channels = mid_channels or round(in_channels * expand_ratio)

        self.inverted_conv = ConvLayer(
            in_channels,
            mid_channels,
            1,
            stride=1,
            norm=norm[0],
            act_func=act_func[0],
            use_bias=use_bias[0],
        )
        self.depth_conv = ConvLayer(
            mid_channels,
            mid_channels,
            kernel_size,
            stride=stride,
            groups=mid_channels,
            norm=norm[1],
            act_func=act_func[1],
            use_bias=use_bias[1],
        )
        self.point_conv = ConvLayer(
            mid_channels,
            out_channels,
            1,
            norm=norm[2],
            act_func=act_func[2],
            use_bias=use_bias[2],
        )

    def forward(self, x: torch.Tensor, valid_mask=None) -> torch.Tensor:
        x = self.inverted_conv(x, valid_mask)
        x = self.depth_conv(x, valid_mask)
        x = self.point_conv(x, valid_mask)
        return x


class FusedMBConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=6,
        groups=1,
        use_bias=False,
        norm=("bn2d", "bn2d"),
        act_func=("relu6", None),
    ):
        super().__init__()
        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        mid_channels = mid_channels or round(in_channels * expand_ratio)

        self.spatial_conv = ConvLayer(
            in_channels,
            mid_channels,
            kernel_size,
            stride,
            groups=groups,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.point_conv = ConvLayer(
            mid_channels,
            out_channels,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
        )

    def forward(self, x: torch.Tensor, valid_mask=None) -> torch.Tensor:
        x = self.spatial_conv(x, valid_mask)
        x = self.point_conv(x, valid_mask)
        return x


class ResBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        mid_channels=None,
        expand_ratio=1,
        use_bias=False,
        norm=("bn2d", "bn2d"),
        act_func=("relu6", None),
    ):
        super().__init__()
        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        mid_channels = mid_channels or round(in_channels * expand_ratio)

        self.conv1 = ConvLayer(
            in_channels,
            mid_channels,
            kernel_size,
            stride,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.conv2 = ConvLayer(
            mid_channels,
            out_channels,
            kernel_size,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
        )

    def forward(self, x: torch.Tensor, valid_mask=None) -> torch.Tensor:
        x = self.conv1(x, valid_mask)
        x = self.conv2(x, valid_mask)
        return x


class LiteMLA(nn.Module):
    r"""Lightweight multi-scale linear attention"""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int or None = None,
        heads_ratio: float = 1.0,
        dim=8,
        use_bias=False,
        norm=(None, "bn2d"),
        act_func=(None, None),
        kernel_func="relu",
        scales: tuple[int, ...] = (5,),
        eps=1.0e-15,
        downsample=None,
    ):
        super(LiteMLA, self).__init__()
        self.eps = eps
        heads = heads or int(in_channels // dim * heads_ratio)

        total_dim = heads * dim

        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        self.dim = dim
        self.qkv = ConvLayer(
            in_channels,
            3 * total_dim,
            1,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.aggreg = nn.ModuleList()

        for scale in scales:
            conv1 = ConvLayer(
                3 * total_dim,
                3 * total_dim,
                kernel_size=scale,
                stride=1,
                groups=3 * total_dim,
                use_bias=use_bias[0],
                norm=None,
                act_func=None,
            ) 
            conv2 = ConvLayer(
                3 * total_dim, 
                3 * total_dim, 
                kernel_size=1,
                stride=1,
                groups=3 * heads, 
                use_bias=use_bias[0],
                norm=None,
                act_func=None,
            )
            self.aggreg.append(nn.ModuleList([conv1, conv2]))

        self.kernel_func = build_act(kernel_func, inplace=False)

        self.proj = ConvLayer(
            total_dim * (1 + len(scales)),
            out_channels,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
        )

        self.bn = nn.BatchNorm2d(self.dim)
        self.act = nn.GELU()
        self.ones_scale1 = nn.Parameter(torch.tensor(1.))
        self.positional_encoding = nn.Parameter(torch.zeros(size=(1, heads*dim*2, 224//downsample, 224//downsample)))
    
    @autocast(enabled=False)
    def qt_attention(self, qkv: torch.Tensor, valid_mask=None) -> torch.Tensor:
        B, _, H, W = list(qkv.size())
        
        if qkv.dtype == torch.float16:
            qkv = qkv.float()
        
        qkv = torch.reshape(qkv, (B, -1, 3 * self.dim, H * W))
        qkv = torch.transpose(qkv, -1, -2)
        q, k, v = (
            qkv[..., 0 : self.dim],
            qkv[..., self.dim : 2 * self.dim],
            qkv[..., 2 * self.dim :],
        )
        
        Bq, Headq, Nq, Cq = q.shape
        
        if valid_mask is not None:
            if valid_mask.shape[2:] != (H, W):
                mask = F.interpolate(valid_mask, size=(H, W), mode='nearest')
            else:
                mask = valid_mask
            
            # Flatten mask and get valid token indices
            mask_flat = mask.flatten(2)  # Shape: (B, 1, H*W)
            valid_indices = mask_flat.squeeze(1).bool()  # Shape: (B, H*W)
            
            # Extract only valid tokens for each batch
            max_valid = valid_indices.sum(dim=1).max().item()
            if max_valid == 0:
                # No valid tokens, return zeros
                out = torch.zeros_like(qkv[..., :self.dim])
                out = torch.transpose(out, -1, -2)
                return torch.reshape(out, (B, -1, H, W))
            
            # Create padded tensors for valid tokens only
            q_valid = torch.zeros(B, Headq, max_valid, Cq, device=q.device, dtype=q.dtype)
            k_valid = torch.zeros(B, Headq, max_valid, Cq, device=k.device, dtype=k.dtype)
            v_valid = torch.zeros(B, Headq, max_valid, Cq, device=v.device, dtype=v.dtype)
            
            # Fill valid tokens
            for b in range(B):
                n_valid = valid_indices[b].sum().item()
                if n_valid > 0:
                    q_valid[b, :, :n_valid] = q[b, :, valid_indices[b]]
                    k_valid[b, :, :n_valid] = k[b, :, valid_indices[b]]
                    v_valid[b, :, :n_valid] = v[b, :, valid_indices[b]]
            
            # Update working variables
            q, k, v = q_valid, k_valid, v_valid
            Nq = max_valid
        
        # Positional encoding (only applied to valid tokens now)
        if H != self.positional_encoding.shape[2] or W != self.positional_encoding.shape[3]:
            absolute_pos_embed = F.interpolate(self.positional_encoding, size=(H, W), mode='bicubic').reshape(-1, Headq, Cq, H*W).transpose(-1,-2)
        else:
            absolute_pos_embed = self.positional_encoding.reshape(-1, Headq, Cq, H*W).transpose(-1,-2)
        
        if valid_mask is not None:
            # Apply positional encoding only to valid positions
            pos_embed_valid = torch.zeros(B, Headq, max_valid, Cq, device=k.device, dtype=k.dtype)
            for b in range(B):
                n_valid = valid_indices[b].sum().item()
                if n_valid > 0:
                    pos_embed_valid[b, :, :n_valid] = absolute_pos_embed[0, :, valid_indices[b]]
            k = k + pos_embed_valid
        else:
            k = k + absolute_pos_embed
        
        # Attention computation (unchanged)
        q = q / (q.norm(dim=-1, keepdim=True) + self.eps)
        k = k / (k.norm(dim=-1, keepdim=True) + self.eps)
        q = q ** 2
        k = k ** 2
        q = q / (q.norm(dim=-1, keepdim=True) + self.eps)
        k = k / (k.norm(dim=-1, keepdim=True) + self.eps)
        
        ones = torch.ones(Bq, Headq, Nq, 1).to(q.device)
        ones1 = ones * self.ones_scale1
        q = torch.cat((q, ones1), dim=-1)
        k = torch.cat((k, ones1), dim=-1)
        
        # Linear matmul
        trans_k = k.transpose(-1, -2)
        v = F.pad(v, (0, 1), mode="constant", value=1)
        kv = torch.matmul(trans_k, v)
        out = torch.matmul(q, kv)
        out = out[..., :-1] / (out[..., -1:] + self.eps)
        
        if valid_mask is not None:
            pass
        else:
            # Original dwconv logic
            num = int(v.shape[2] ** 0.5)
            e = v.shape[1]
            feature_map = rearrange(v, "b e (w h) c -> (b e) c w h", w=num, h=num)
            feature_map = rearrange(self.act(self.bn(feature_map[:,:-1,:,:])), "(b e) c w h -> b e (w h) c", e=e)
            out = out + feature_map
        
        # Restore original spatial structure
        if valid_mask is not None:
            final_out = torch.zeros(B, Headq, H*W, Cq, device=out.device, dtype=out.dtype)
            for b in range(B):
                n_valid = valid_indices[b].sum().item()
                if n_valid > 0:
                    final_out[b, :, valid_indices[b]] = out[b, :, :n_valid]
            out = final_out
        
        out = torch.transpose(out, -1, -2)
        out = torch.reshape(out, (B, -1, H, W))
        return out

    # @autocast(enabled=False)
    # def qt_attention(self, qkv: torch.Tensor, valid_mask) -> torch.Tensor:
    #     B, _, H, W = list(qkv.size())

    #     if qkv.dtype == torch.float16:
    #         qkv = qkv.float()

    #     qkv = torch.reshape(
    #         qkv,
    #         (
    #             B,
    #             -1,
    #             3 * self.dim,
    #             H * W,
    #         ),
    #     )
    #     qkv = torch.transpose(qkv, -1, -2)
    #     q, k, v = (
    #         qkv[..., 0 : self.dim],
    #         qkv[..., self.dim : 2 * self.dim],
    #         qkv[..., 2 * self.dim :],
    #     )

    #     Bq, Headq, Nq, Cq = q.shape
    #     if H != self.positional_encoding.shape[2] or W != self.positional_encoding.shape[3]:
    #         absolute_pos_embed = F.interpolate(self.positional_encoding, size=(H, W), mode='bicubic').reshape(-1, Headq, Cq, H*W).transpose(-1,-2)
    #     else:
    #         absolute_pos_embed = self.positional_encoding.reshape(-1, Headq, Cq, H*W).transpose(-1,-2)

    #     k = k + absolute_pos_embed

    #     q = q / (q.norm(dim=-1, keepdim=True) + self.eps)
    #     k = k / (k.norm(dim=-1, keepdim=True) + self.eps)
    #     q = q ** 2
    #     k = k ** 2
    #     q = q / (q.norm(dim=-1, keepdim=True) + self.eps)
    #     k = k / (k.norm(dim=-1, keepdim=True) + self.eps)

    #     ones = torch.ones(Bq,Headq,Nq,1).to(q.device)
    #     ones1 = ones * self.ones_scale1
    #     q = torch.cat((q, ones1), dim=-1)
    #     k = torch.cat((k, ones1), dim=-1)
        
    #     # linear matmul
    #     trans_k = k.transpose(-1, -2)

    #     v = F.pad(v, (0, 1), mode="constant", value=1)
    #     kv = torch.matmul(trans_k, v)
    #     out = torch.matmul(q, kv)
    #     out = out[..., :-1] / (out[..., -1:] + self.eps)

    #     ############# add dwconv
    #     num = int(v.shape[2] ** 0.5)
    #     e = v.shape[1]
    #     feature_map = rearrange(v, "b e (w h) c -> (b e) c w h", w=num, h=num)
    #     feature_map = rearrange(self.act(self.bn(feature_map[:,:-1,:,:])), "(b e) c w h -> b e (w h) c", e=e)
    #     out = out + feature_map
    #     #############
        
    #     out = torch.transpose(out, -1, -2)
    #     out = torch.reshape(out, (B, -1, H, W))
    #     return out

    def forward(self, x: torch.Tensor, valid_mask=None) -> torch.Tensor:
        # generate multi-scale q, k, v
        qkv = self.qkv(x, valid_mask)
        multi_scale_qkv = [qkv]
        
        for conv1, conv2 in self.aggreg:
            tmp = conv1(qkv, valid_mask)
            tmp = conv2(tmp, valid_mask)
            multi_scale_qkv.append(tmp)

        multi_scale_qkv = torch.cat(multi_scale_qkv, dim=1)

        # out = self.relu_linear_att(multi_scale_qkv)
        # out = self.softmax_att(multi_scale_qkv)
        out = self.qt_attention(multi_scale_qkv, valid_mask)
        out = self.proj(out, valid_mask)

        return out

    @staticmethod
    def configure_litemla(model: nn.Module, **kwargs) -> None:
        eps = kwargs.get("eps", None)
        for m in model.modules():
            if isinstance(m, LiteMLA):
                if eps is not None:
                    m.eps = eps



class EfficientViTBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        heads_ratio: float = 1.0,
        dim=32,
        expand_ratio: float = 4,
        norm="bn2d",
        act_func="hswish",
        downsample=None,
    ):
        super(EfficientViTBlock, self).__init__()
        self.context_module = ResidualBlock(
            LiteMLA(
                in_channels=in_channels,
                out_channels=in_channels,
                heads_ratio=heads_ratio,
                dim=dim,
                norm=(None, norm),
                downsample=downsample,
            ),
            IdentityLayer()#, post_norm=nn.BatchNorm2d(in_channels)
        )
        local_module = MBConv(
            in_channels=in_channels,
            out_channels=in_channels,
            expand_ratio=expand_ratio,
            use_bias=(True, True, False),
            norm=(None, None, norm),
            act_func=(act_func, act_func, None),
        )
        self.local_module = ResidualBlock(local_module, IdentityLayer())#,post_norm=nn.BatchNorm2d(in_channels))

    def forward(self, x: torch.Tensor, valid_mask=None) -> torch.Tensor:
        x = self.context_module(x, valid_mask)
        x = self.local_module(x, valid_mask)
        return x # ori or deepnorm_nlp


#################################################################################
#                             Functional Blocks                                 #
#################################################################################


class ResidualBlock(nn.Module): # ori
    def __init__(
        self,
        main: nn.Module or None,
        shortcut: nn.Module or None,
        post_act=None,
        pre_norm: nn.Module or None = None,
    ):
        super(ResidualBlock, self).__init__()

        self.pre_norm = pre_norm
        self.main = main
        self.shortcut = shortcut
        self.post_act = build_act(post_act)

    def forward_main(self, x: torch.Tensor, valid_mask) -> torch.Tensor:
        if self.pre_norm is None:
            return self.main(x, valid_mask)
        else:
            return self.main(self.pre_norm(x), valid_mask)

    def forward(self, x: torch.Tensor, valid_mask=None) -> torch.Tensor:
        if self.main is None:
            res = x
        elif self.shortcut is None:
            res = self.forward_main(x, valid_mask)
        else:
            res = self.forward_main(x, valid_mask) + self.shortcut(x, valid_mask)
            if self.post_act:
                res = self.post_act(res)
        return res

class DAGBlock(nn.Module):
    def __init__(
        self,
        inputs: dict[str, nn.Module],
        merge: str,
        post_input: nn.Module or None,
        middle: nn.Module,
        outputs: dict[str, nn.Module],
    ):
        super(DAGBlock, self).__init__()

        self.input_keys = list(inputs.keys())
        self.input_ops = nn.ModuleList(list(inputs.values()))
        self.merge = merge
        self.post_input = post_input

        self.middle = middle

        self.output_keys = list(outputs.keys())
        self.output_ops = nn.ModuleList(list(outputs.values()))

    def forward(self, feature_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        feat = [op(feature_dict[key]) for key, op in zip(self.input_keys, self.input_ops)]
        if self.merge == "add":
            feat = list_sum(feat)
        elif self.merge == "cat":
            feat = torch.concat(feat, dim=1)
        else:
            raise NotImplementedError
        if self.post_input is not None:
            feat = self.post_input(feat)
        feat = self.middle(feat)
        for key, op in zip(self.output_keys, self.output_ops):
            feature_dict[key] = op(feat)
        return feature_dict


class OpSequential(nn.Module):
    def __init__(self, op_list: list[nn.Module or None]):
        super(OpSequential, self).__init__()
        valid_op_list = []
        for op in op_list:
            if op is not None:
                valid_op_list.append(op)
        self.op_list = nn.ModuleList(valid_op_list)

    def forward(self, x: torch.Tensor, valid_mask=None) -> torch.Tensor:
        for op in self.op_list:
            x = op(x, valid_mask)
        return x
