"""
Pure-PyTorch Swin Transformer matching MMPretrain/mmcv architecture EXACTLY.

Parameter names match MMPretrain checkpoint keys so NO key mapping is
needed — just load_state_dict() directly.

Checkpoint key structure:
    backbone.patch_embed.projection.weight
    backbone.patch_embed.norm.weight
    backbone.stages.X.blocks.Y.norm1.weight
    backbone.stages.X.blocks.Y.attn.w_msa.qkv.weight
    backbone.stages.X.blocks.Y.attn.w_msa.proj.weight
    backbone.stages.X.blocks.Y.attn.w_msa.relative_position_bias_table
    backbone.stages.X.blocks.Y.ffn.layers.0.0.weight   (fc1)
    backbone.stages.X.blocks.Y.ffn.layers.1.weight      (fc2)
    backbone.stages.X.downsample.norm.weight
    backbone.stages.X.downsample.reduction.weight
    backbone.norm3.weight
    head.fc.weight

CRITICAL NOTES on matching mmcv behavior:
  1. PatchMerging uses nn.Unfold (same as mmcv) — the channel ordering of
     nn.Unfold differs from manual x[:,0::2,0::2] slicing, which would
     produce garbled output with weights trained under mmcv.
  2. relative_position_index is computed using MMPretrain's double_step_seq
     method with .flip(1), which may differ from the standard meshgrid approach.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── Helpers ─────────────────────────────────────────

def _window_partition(x, ws):
    """(B, H, W, C) → (B*nW, ws, ws, C)"""
    B, H, W, C = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, ws, ws, C)


def _window_reverse(w, ws, H, W):
    """(B*nW, ws, ws, C) → (B, H, W, C)"""
    B = int(w.shape[0] / (H * W / ws / ws))
    x = w.view(B, H // ws, W // ws, ws, ws, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)


def _double_step_seq(step1, len1, step2, len2):
    """Reproduce MMPretrain's WindowMSA.double_step_seq exactly."""
    seq1 = torch.arange(0, step1 * len1, step1)
    seq2 = torch.arange(0, step2 * len2, step2)
    return (seq1[:, None] + seq2[None, :]).reshape(1, -1)


# ── Attention ───────────────────────────────────────

class WindowMSA(nn.Module):
    """Window Multi-head Self-Attention — matches MMPretrain's w_msa."""

    def __init__(self, dim, num_heads, ws=7):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = (ws, ws)

        self.qkv = nn.Linear(dim, dim * 3, bias=True)
        self.proj = nn.Linear(dim, dim)

        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * ws - 1) * (2 * ws - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        # Match MMPretrain's double_step_seq exactly
        Wh, Ww = ws, ws
        rel_index_coords = _double_step_seq(2 * Ww - 1, Wh, 1, Ww)
        rel_position_index = rel_index_coords + rel_index_coords.T
        rel_position_index = rel_position_index.flip(1).contiguous()
        self.register_buffer("relative_position_index", rel_position_index)

    def forward(self, x, mask=None):
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = q * self.scale
        attn = q @ k.transpose(-2, -1)

        Wh, Ww = self.window_size
        bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(Wh * Ww, Wh * Ww, -1).permute(2, 0, 1).contiguous()
        attn = attn + bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)

        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        return self.proj(x)


class ShiftWindowMSA(nn.Module):
    """Shifted-Window MSA wrapper — matches MMPretrain's attn.

    Includes pad_small_map=False behavior (classification mode):
    when feature map equals window size, disable shifting.
    """

    def __init__(self, dim, num_heads, ws=7, shift=False):
        super().__init__()
        self.w_msa = WindowMSA(dim, num_heads, ws)
        self.window_size = ws
        self.shift_size = ws // 2 if shift else 0

    def forward(self, x, hw):
        B, L, C = x.shape
        H, W = hw
        x = x.view(B, H, W, C)

        window_size = self.window_size
        shift_size = self.shift_size

        # Match MMPretrain: if feature map == window size, disable shift
        if min(H, W) == window_size:
            shift_size = 0

        # Pad feature maps to multiples of window_size (matches mmcv)
        pad_r = (window_size - W % window_size) % window_size
        pad_b = (window_size - H % window_size) % window_size
        if pad_r > 0 or pad_b > 0:
            x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))
        H_pad, W_pad = x.shape[1], x.shape[2]

        # Cyclic shift
        if shift_size > 0:
            sx = torch.roll(x, (-shift_size, -shift_size), (1, 2))
            amask = self._get_attn_mask(
                (H_pad, W_pad), window_size, shift_size, x.device)
        else:
            sx = x
            amask = None

        # Window partition → attention → reverse
        wins = _window_partition(sx, window_size).view(
            -1, window_size * window_size, C)
        wins = self.w_msa(wins, amask)
        wins = wins.view(-1, window_size, window_size, C)
        sx = _window_reverse(wins, window_size, H_pad, W_pad)

        # Reverse cyclic shift
        if shift_size > 0:
            x = torch.roll(sx, (shift_size, shift_size), (1, 2))
        else:
            x = sx

        # Remove padding
        if pad_r > 0 or pad_b > 0:
            x = x[:, :H, :W, :].contiguous()

        return x.view(B, H * W, C)

    @staticmethod
    def _get_attn_mask(hw_shape, window_size, shift_size, device):
        """Generate attention mask — matches MMPretrain's get_attn_mask."""
        if shift_size > 0:
            H, W = hw_shape
            img_mask = torch.zeros(1, H, W, 1, device=device)
            h_slices = (
                slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None),
            )
            w_slices = (
                slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None),
            )
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = _window_partition(img_mask, window_size)
            mask_windows = mask_windows.view(-1, window_size * window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0)
            attn_mask = attn_mask.masked_fill(attn_mask == 0, 0.0)
            return attn_mask
        return None


# ── Block & Stage ───────────────────────────────────

class SwinBlock(nn.Module):
    def __init__(self, dim, heads, ws=7, shift=False, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = ShiftWindowMSA(dim, heads, ws, shift)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        # FFN structure matches mmcv's FFN:
        #   layers[0] = Sequential(Linear, GELU, Dropout(0))
        #   layers[1] = Linear
        #   layers[2] = Dropout(0)
        # Dropout(0) is identity in eval, but we include it for key compat.
        self.ffn = nn.Module()
        self.ffn.layers = nn.Sequential(
            nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Dropout(0.0)),
            nn.Linear(hidden, dim),
            nn.Dropout(0.0),
        )

    def forward(self, x, hw):
        identity = x
        x = self.norm1(x)
        x = self.attn(x, hw)
        x = x + identity

        identity = x
        x = self.norm2(x)
        x = self.ffn.layers(x)
        x = x + identity
        return x


class PatchMerging(nn.Module):
    """Patch merging layer — matches mmcv's PatchMerging exactly.

    Uses nn.Unfold to merge patches, matching the channel ordering that
    the checkpoint weights were trained with. Manual slicing (x[:,0::2,0::2])
    produces DIFFERENT channel ordering and corrupts inference.
    """

    def __init__(self, dim):
        super().__init__()
        self.in_channels = dim
        self.out_channels = 2 * dim
        # mmcv uses nn.Unfold with kernel_size=2, stride=2
        self.sampler = nn.Unfold(
            kernel_size=2, dilation=1, padding=0, stride=2)
        sample_dim = 4 * dim  # 2*2 * dim
        self.norm = nn.LayerNorm(sample_dim)
        self.reduction = nn.Linear(sample_dim, 2 * dim, bias=False)

    def forward(self, x, hw):
        B, L, C = x.shape
        H, W = hw
        assert L == H * W

        # Reshape to image format for Unfold: (B, C, H, W)
        x = x.view(B, H, W, C).permute(0, 3, 1, 2)  # B, C, H, W
        x = self.sampler(x)  # B, 4*C, (H/2)*(W/2)

        # kernel_size=2, stride=2, padding=0, dilation=1 → output = H//2, W//2
        out_h = H // 2
        out_w = W // 2

        x = x.transpose(1, 2)  # B, (H/2)*(W/2), 4*C
        x = self.norm(x)
        x = self.reduction(x)
        return x, (out_h, out_w)


class SwinStage(nn.Module):
    def __init__(self, dim, depth, heads, ws=7, mlp_ratio=4.0, downsample=True):
        super().__init__()
        self.blocks = nn.ModuleList([
            SwinBlock(dim, heads, ws, shift=(i % 2 == 1), mlp_ratio=mlp_ratio)
            for i in range(depth)
        ])
        self.downsample = PatchMerging(dim) if downsample else None

    def forward(self, x, hw):
        for blk in self.blocks:
            x = blk(x, hw)
        if self.downsample is not None:
            x, hw = self.downsample(x, hw)
        return x, hw


# ── Full Model ──────────────────────────────────────

class PatchEmbed(nn.Module):
    def __init__(self, in_ch=3, dim=128, ps=4):
        super().__init__()
        self.projection = nn.Conv2d(in_ch, dim, ps, ps)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        x = self.projection(x)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, (H, W)


class SwinBackbone(nn.Module):
    def __init__(self, dim=128, depths=(2, 2, 18, 2),
                 heads=(4, 8, 16, 32), ws=7, mlp_ratio=4.0):
        super().__init__()
        self.patch_embed = PatchEmbed(3, dim, 4)
        self.stages = nn.ModuleList()
        d = dim
        for i, (dep, h) in enumerate(zip(depths, heads)):
            self.stages.append(SwinStage(
                d, dep, h, ws, mlp_ratio, downsample=(i < len(depths) - 1)
            ))
            if i < len(depths) - 1:
                d *= 2
        self.norm3 = nn.LayerNorm(d)
        self.num_features = d

    def forward(self, x):
        x, hw = self.patch_embed(x)
        for s in self.stages:
            x, hw = s(x, hw)
        x = self.norm3(x)
        return x.mean(dim=1)


class SwinClassifier(nn.Module):
    """Swin-B classifier — keys match MMPretrain checkpoint exactly."""

    def __init__(self, num_classes=200):
        super().__init__()
        self.backbone = SwinBackbone()
        # Use a Module wrapper so key is head.fc.weight (not head.weight)
        self.head = nn.Module()
        self.head.fc = nn.Linear(self.backbone.num_features, num_classes)

    def forward(self, x):
        return self.head.fc(self.backbone(x))
