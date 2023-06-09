#!/usr/bin/env python3

# Copyright (c) 2022, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
# written by Ali Hatamizadeh and Pavlo Molchanov from nvResearch
# tweaked by Pavlo Molchanov

import torch
import torch.nn as nn
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_, DropPath, to_2tuple, LayerNorm2d
import pdb
from models.others.positional_encodding import PosEmbMLPSwinv2D, PosEmbMLPSwinv1D
from models.layers.extra import GaussianDropout

def _to_channel_last(x):
    """
    Args:
        x: (B, C, H, W)

    Returns:
        x: (B, H, W, C)
    """
    return x.permute(0, 2, 3, 1)


def _to_channel_first(x):
    """
    Args:
        x: (B, H, W, C)

    Returns:
        x: (B, C, H, W)
    """
    return x.permute(0, 3, 1, 2)


def window_partition(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size: window size
        h_w: Height of window
        w_w: Width of window
    Returns:
        local window features (num_windows*B, window_size*window_size, C)
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, window_size*window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: local window features (num_windows*B, window_size, window_size, C)
        window_size: Window size
        H: Height of image
        W: Width of image
    Returns:
        x: (B, C, H, W)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(B,windows.shape[2], H, W)
    return x


class Mlp(nn.Module):
    """
    Multi-Layer Perceptron (MLP) block
    """

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        """
        Args:
            in_features: input features dimension.
            hidden_features: hidden features dimension.
            out_features: output features dimension.
            act_layer: activation function.
            drop: dropout rate.
        """

        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = GaussianDropout(drop)

    def forward(self, x):
        x_size = x.size()
        x = x.view(-1, x_size[-1])
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = x.view(x_size)
        return x

class Downsample(nn.Module):
    """
    Down-sampling block based on: "Hatamizadeh et al.,
    Global Context Vision Transformers <https://arxiv.org/abs/2206.09959>"
    """

    def __init__(self,
                 dim,
                 keep_dim=False,
                 ):
        """
        Args:
            dim: feature size dimension.
            norm_layer: normalization layer.
            keep_dim: bool argument for maintaining the resolution.
        """

        super().__init__()
        if keep_dim:
            dim_out = dim
        else:
            dim_out = 2 * dim
        self.norm = LayerNorm2d(dim)
        self.reduction = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, 2, 1, bias=False),
        )


    def forward(self, x):
        x = self.norm(x)
        x = self.reduction(x)
        return x


class PatchEmbed(nn.Module):
    """
    Patch embedding block based on: "Hatamizadeh et al.,
    Global Context Vision Transformers <https://arxiv.org/abs/2206.09959>"
    """

    def __init__(self, in_chans=3, in_dim=64, dim=96, simple_stem=True):
        """
        Args:
            in_chans: number of input channels.
            dim: feature size dimension.
        """

        super().__init__()
        if simple_stem:
            self.proj = nn.Conv2d(in_chans, in_dim, 3, 2, 1, bias=False)
            self.conv_down = nn.Sequential(
                nn.Conv2d(in_dim, dim, 3, 2, 1, bias=False),
                nn.BatchNorm2d(dim, eps=1e-4),
                nn.GELU()
            )
        else:
            self.proj = nn.Identity()
            self.conv_down = nn.Sequential(
                nn.Conv2d(in_chans, in_dim, 3, 2, 1, bias=False),
                nn.BatchNorm2d(in_dim, eps=1e-4),
                nn.ReLU(), # relu?
                nn.Conv2d(in_dim, dim, 3, 2, 1, bias=False),
                nn.BatchNorm2d(dim, eps=1e-4),
                nn.ReLU()
            )

    def forward(self, x):
        x = self.proj(x)
        x = self.conv_down(x)
        return x



class ConvBlock(nn.Module):

    def __init__(self, dim,
                 drop_path=0.,
                 layer_scale=None,
                 kernel_size=3):
        super().__init__()

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(dim, eps=1e-5)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(dim, eps=1e-5)
        self.layer_scale = layer_scale
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.gamma = nn.Parameter(layer_scale * torch.ones(dim))
            self.layer_scale = True
        else:
            self.layer_scale = False
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x, global_feature=None):
        input = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        if self.layer_scale:
            x = x * self.gamma.view(1, -1, 1, 1)
        x = input + self.drop_path(x)
        return x, global_feature


class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., resolution=0,
                 seq_length=0, annealing=True):
        # taken from EdgeViT and tweaked with attention bias.
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.head_dim = dim // num_heads

        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = GaussianDropout(attn_drop, annealing=annealing)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = GaussianDropout(proj_drop)

        # attention positional bias
        self.pos_emb_funct = PosEmbMLPSwinv2D(window_size=[resolution, resolution],
                                              pretrained_window_size=[resolution, resolution],
                                              num_heads=num_heads,
                                              seq_length=seq_length)

        self.resolution = resolution

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale

        # positional bias
        attn = self.pos_emb_funct(attn, self.resolution ** 2)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, -1, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class SelfAttnHAT(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1., window_size=7, last=False, layer_scale=None, cascade=0,
                 extra_ln=False, annealing=True):
        super().__init__()
        # positional encoding for windowed attention tokens
        self.pos_embed = PosEmbMLPSwinv1D(dim, rank=2, seq_length=window_size**2)
        self.norm1 = norm_layer(dim)
        # number of carrier tokens per every window
        cr_tokens_total = 2**(2**cascade) if sr_ratio > 1 else 0
        self.cr_window = 2**cascade

        self.attn = WindowAttention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, annealing=annealing, proj_drop=drop, resolution=window_size,
            seq_length=window_size**2 + cr_tokens_total) # plus carrier tokens

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.window_size = window_size

        use_layer_scale = layer_scale is not None and type(layer_scale) in [int, float]
        self.gamma3 = nn.Parameter(layer_scale * torch.ones(dim))  if use_layer_scale else 1
        self.gamma4 = nn.Parameter(layer_scale * torch.ones(dim))  if use_layer_scale else 1

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            # if do hierarchical attention, this part is for carrier tokens
            self.hat_norm1 = norm_layer(dim)
            self.hat_norm2 = norm_layer(dim)
            self.hat_attn = WindowAttention(
                dim,
                num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                attn_drop=attn_drop, proj_drop=drop, resolution=cr_tokens_total,
                seq_length=cr_tokens_total**2)

            self.hat_mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
            self.hat_drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
            self.hat_pos_embed = PosEmbMLPSwinv1D(dim, rank=2, seq_length=cr_tokens_total**2)
            self.gamma1 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1
            self.gamma2 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1

            self.upsampler = nn.Upsample(size=window_size, mode='nearest')

        self.last = last # keep track for the last bloxk to explicitely add carrier tokens to feature maps

        self.extra_ln = extra_ln
        if extra_ln:
            self.extra_ln = nn.LayerNorm(dim)

    def forward(self, x, carrier_tokens):
        B, T, N = x.shape
        ct = carrier_tokens

        x = self.pos_embed(x)

        if self.sr_ratio > 1:
            # do hierarchical attention via carrier tokens
            # first do attention for carreir tokens
            Bg, Ng, Hg = ct.shape

            # positional bias for carrier tokens
            ct = self.hat_pos_embed(ct)

            # attention plus mlp
            ct = ct + self.hat_drop_path(self.gamma1*self.hat_attn(self.hat_norm1(ct)))
            ct = ct + self.hat_drop_path(self.gamma2*self.hat_mlp(self.hat_norm2(ct)))

            ct = ct.reshape(x.shape[0], -1, N)
            # concatenate carrier_tokens to the windowed tokens
            x = torch.cat((ct, x), dim=1)

        # window attention together with carrier tokens
        x = x + self.drop_path(self.gamma3*self.attn(self.norm1(x)))
        x = x + self.drop_path(self.gamma4*self.mlp(self.norm2(x)))

        if self.sr_ratio > 1:
            # for hierarchical attention we need to split carrier tokens and window tokens back
            ctr, x = x.split([x.shape[1]-self.window_size*self.window_size, self.window_size*self.window_size], dim=1)
            ct = ctr.reshape(Bg, Ng, Hg) # reshape carrier tokens.
            # For attention, we were only adding corresponding tokens to the window tokens
            # however, for self attention ct need to be reshape such that we do attention over all cr

            if self.last and 1:
                # add carrier token information into the image
                # interpolate once in the end of the block
                # x = x + self.gamma1*torch.nn.functional.interpolate(ctr.transpose(1, 2), size=(x.shape[1])).transpose(1, 2)
                ctr_image_space = ctr.transpose(1, 2).reshape(B, N, self.cr_window, self.cr_window)
                x = x + self.gamma1 * self.upsampler(ctr_image_space.to(dtype=torch.float32)).flatten(2).transpose(1,2).to(dtype=x.dtype)

        if self.extra_ln:
            x = self.extra_ln(x)

        return x, ct



class GlobalTokenizer(nn.Module):
    """
    Global tokenizer based on: "Hatamizadeh et al.,
    FasterViT: Lightning-Fast Vision Transformers with Scaled-up Resolution
    """

    def __init__(self,
                 dim,
                 input_resolution,
                 window_size,
                 cascade=0):
        """
        Args:
            dim: feature size dimension.
            input_resolution: input image resolution.
            window_size: window size.
            cascade: spatial dimension of carrier token local window
        """
        super().__init__()

        output_size = int((2 ** cascade) * input_resolution/window_size)
        stride_size = int(input_resolution/output_size)
        kernel_size = input_resolution - (output_size - 1) * stride_size

        # from Twins, EdgeViT, PEG, pos encoding is done by linear conv2d
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

        to_global_feature = nn.Sequential()
        to_global_feature.add_module("pos", self.pos_embed)
        to_global_feature.add_module("pool", nn.AvgPool2d(kernel_size=kernel_size,
                                         stride=stride_size))
        self.to_global_feature = to_global_feature

    def forward(self, x):
        x = self.to_global_feature(x)
        x = x.flatten(2).transpose(1, 2) #to channel last
        return x


class FasterViTLayer(nn.Module):
    """
    GCViT layer based on: "Hatamizadeh et al.,
    Global Context Vision Transformers <https://arxiv.org/abs/2206.09959>"
    """

    def __init__(self,
                 dim,
                 depth,
                 input_resolution,
                 num_heads,
                 window_size,
                 cascade=0,
                 conv=False,
                 downsample=True,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 annealing=True,
                 drop_path=0.,
                 layer_scale=None,
                 layer_scale_conv=None,
                 only_local=False,
                 hierarchy=True,
                 intermediate_ln_every = 0,

    ):
        """
        Args:
            dim: feature size dimension.
            depth: number of layers in each stage.
            input_resolution: input image resolution.
            window_size: window size in each stage.
            downsample: bool argument for down-sampling.
            mlp_ratio: MLP ratio.
            num_heads: number of heads in each stage.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop: dropout rate.
            attn_drop: attention dropout rate.
            drop_path: drop path rate.
            norm_layer: normalization layer.
            layer_scale: layer scaling coefficient.
        """

        super().__init__()
        self.conv = conv
        self.transformer_block = False
        if conv:
            self.blocks = nn.ModuleList([
                ConvBlock(dim=dim,
                          drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                          layer_scale=layer_scale_conv)
                for i in range(depth)])
            self.transformer_block = False
        else:
            # channel last is expected at output
            sr_ratio = input_resolution // window_size if not only_local else 1
            self.blocks = nn.ModuleList([
                SelfAttnHAT(dim=dim,
                            num_heads=num_heads,
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            drop=drop,
                            attn_drop=attn_drop,
                            annealing=annealing,
                            drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                            sr_ratio= sr_ratio,
                            window_size=window_size,
                            last=(i==depth-1),
                            layer_scale=layer_scale,
                            cascade=cascade,
                            extra_ln=(intermediate_ln_every and (1 + i) % intermediate_ln_every == 0),
                           )
                for i in range(depth)])
            self.transformer_block = True

        self.downsample = None if not downsample else Downsample(dim=dim)

        if len(self.blocks) and not only_local and input_resolution // window_size > 1 and hierarchy and not self.conv:
            self.global_tokenizer = GlobalTokenizer(dim, input_resolution, window_size, cascade=cascade)
            self.do_gt = True
            print("Global Tokens Generation block is initialized")

        else:
            self.do_gt = False

        self.window_size = window_size


    def forward(self, x):
        global_feature = self.global_tokenizer(x) if self.do_gt else None
        B, C, H, W = x.shape
        if self.transformer_block:
            # window feature maps
            x = window_partition(x, self.window_size)

        for blk in self.blocks:
            x, global_feature = blk(x, global_feature)


        if self.transformer_block:
            # reshape local windows into global feature map again
            x = window_reverse(x, self.window_size, H, W)

        if self.downsample is None:
            return x

        return self.downsample(x)


class FasterViT(nn.Module):
    """
    FasterViT based on: "Hatamizadeh et al.,
    """

    def __init__(self,
                 dim,
                 in_dim,
                 depths,
                 window_size,
                 cascade,
                 mlp_ratio,
                 num_heads,
                 resolution=224,
                 drop_path_rate=0.2,
                 in_chans=3,
                 num_classes=1000,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 layer_scale=None,
                 layer_scale_conv=None,
                 layer_norm_last=False,
                 simple_stem=True,
                 only_local_attention=False,
                 intermediate_ln_every=0,
                 annealing=True,
                 **kwargs):
        """
        Args:
            dim: feature size dimension.
            depths: number of layers in each stage.
            window_size: window size in each stage.
            mlp_ratio: MLP ratio.
            num_heads: number of heads in each stage.
            resolution: input image resolution.
            drop_path_rate: drop path rate.
            in_chans: number of input channels.
            num_classes: number of classes.
            qkv_bias: bool argument for query, key, value learnable bias.
            qk_scale: bool argument to scaling query, key.
            drop_rate: dropout rate.
            attn_drop_rate: attention dropout rate.
            norm_layer: normalization layer.
            layer_scale: layer scaling coefficient.
        """
        super().__init__()

        num_features = int(dim * 2 ** (len(depths) - 1))
        self.num_classes = num_classes
        self.patch_embed = PatchEmbed(in_chans=in_chans, in_dim=in_dim, dim=dim, simple_stem=simple_stem)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.levels = nn.ModuleList()
        for i in range(len(depths)):
            conv = True if (i == 0 or i == 1) else False

            level = FasterViTLayer(dim=int(dim * 2 ** i),
                                   depth=depths[i],
                                   num_heads=num_heads[i],
                                   window_size=window_size[i],
                                   cascade=cascade,
                                   mlp_ratio=mlp_ratio,
                                   qkv_bias=qkv_bias,
                                   qk_scale=qk_scale,
                                   conv=conv,
                                   drop=drop_rate,
                                   attn_drop=attn_drop_rate,
                                   annealing=annealing,
                                   drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
                                   downsample=(i < 3),
                                   layer_scale=layer_scale,
                                   layer_scale_conv=layer_scale_conv,
                                   input_resolution=int(2 ** (-2 - i) * resolution),
                                   only_local=only_local_attention,
                                   intermediate_ln_every=intermediate_ln_every)

            self.levels.append(level)

        self.norm = LayerNorm2d(num_features) if layer_norm_last else nn.BatchNorm2d(num_features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, LayerNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'rpb'}

    def forward_features(self, x):
        x = self.patch_embed(x)
        for level in self.levels:
            x = level(x)
        x = self.norm(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x


@register_model
def faster_vit_0_224(pretrained=False, **kwargs):
    model = FasterViT(depths=[2, 3, 6, 5],
                      num_heads=[2, 4, 8, 16],
                      window_size=[8, 8, 7, 7],
                      cascade=1,
                      dim=64,
                      in_dim=64,
                      mlp_ratio=4,
                      resolution=224,
                      drop_path_rate=0.2,
                      simple_stem=True,
                      **kwargs)

    if pretrained:
        model.load_state_dict(torch.load(pretrained))
    return model


@register_model
def faster_vit_1_224(pretrained=False, **kwargs):
    model = FasterViT(depths=[1, 3, 8, 5],
                      num_heads=[2, 4, 8, 16],
                      window_size=[8, 8, 7, 7],
                      cascade=1,
                      dim=80,
                      in_dim=32,
                      mlp_ratio=4,
                      resolution=224,
                      drop_path_rate=0.2,
                      simple_stem=True,
                      **kwargs)

    if pretrained:
        model.load_state_dict(torch.load(pretrained))
    return model


@register_model
def faster_vit_2_224(pretrained=False, **kwargs):
    model = FasterViT(depths=[3, 3, 8, 5],
                      num_heads=[2, 4, 8, 16],
                      window_size=[8, 8, 7, 7],
                      cascade=1,
                      dim=96,
                      in_dim=64,
                      mlp_ratio=4,
                      resolution=224,
                      drop_path_rate=0.2,
                      layer_scale=1e-5,
                      layer_scale_conv=None,
                      simple_stem=True,
                      **kwargs)

    if pretrained:
        model.load_state_dict(torch.load(pretrained))
    return model


@register_model
def faster_vit_3_224(pretrained=False, **kwargs):
    model = FasterViT(depths=[3, 3, 12, 5],
                      num_heads=[2, 4, 8, 16],
                      window_size=[7, 7, 7, 7],
                      cascade=1,
                      dim=128,
                      in_dim=64,
                      mlp_ratio=4,
                      resolution=224,
                      drop_path_rate=0.3,
                      layer_scale=1e-5,
                      layer_scale_conv=None,
                      simple_stem=True,
                      **kwargs)

    if pretrained:
        model.load_state_dict(torch.load(pretrained))
    return model


@register_model
def faster_vit_4_224(pretrained=False, **kwargs):
    model = FasterViT(depths=[3, 3, 12, 5],
                      num_heads=[4, 8, 16, 32],
                      window_size=[7, 7, 7, 7],
                      cascade=1,
                      dim=196,
                      in_dim=64,
                      mlp_ratio=4,
                      resolution=224,
                      drop_path_rate=0.5,
                      layer_scale=1e-5,
                      layer_scale_conv=None,
                      layer_norm_last=False,
                      simple_stem=True,
                      **kwargs)

    if pretrained:
        model.load_state_dict(torch.load(pretrained))
    return model


if __name__ == "__main__":
    print("Testing model")
    from ptflops import get_model_complexity_info
    from torch.profiler import profile, record_function, ProfilerActivity
    model = faster_vit_4_224()
    channel_last = True
    compute_latency = False
    bs = 64
    resolution = 224
    input_data = torch.randn((bs, 3, resolution, resolution), device='cuda').cuda()
    if channel_last:
        input_data = input_data.to(memory_format=torch.channels_last)
        model = model.to(memory_format=torch.channels_last)

    model.cuda()
    model.eval()
    output = model(input_data)

    # train check
    loss = output.pow(2).sum()
    loss.backward()
    print(output.shape)

    if 1:
        for module in model.modules():
            if hasattr(module, 'switch_to_deploy'):
                module.switch_to_deploy()


    macs, params = get_model_complexity_info(model, tuple([3, resolution, resolution]),
                                             as_strings=False, print_per_layer_stat=False, verbose=False)

    print(f"Model stats: macs: {macs}, and params: {params}")

    if compute_latency:
        from models.latencyestimator import compute_latency_trt
        compute_latency_trt(model, 3, 224, 224, bs, enable_fp16=True)

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
        with record_function("model_inference"):
            with torch.cuda.amp.autocast():
                output = model(input_data)

    prof.export_chrome_trace("trace.json")
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    # warm up
    with torch.cuda.amp.autocast():
        for ii in range(100):
            output = model(input_data)

    # speed
    import time

    start_time = time.time()
    with torch.cuda.amp.autocast():
        for ii in range(100):
            output = model(input_data)
        torch.cuda.synchronize()
    end_time = time.time()
    print(f"Throughput {bs * 1.0 / ((end_time - start_time) / 100)}")

# if __name__ == "__main__":
#     print("Testing model")
#     from ptflops import get_model_complexity_info
#     from torch.profiler import profile, record_function, ProfilerActivity
#
#     import argparse
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--trt", help="run trt mode",
#                         action="store_true")
#     parser.add_argument("--sparse", help="run with sparse mode",
#                         action="store_true")
#     args = parser.parse_args()
#
#
#     model = faster_vit_3_224_edge_hat2()
#
#
#     channel_last = True
#     compute_latency = False
#     bs = 64
#     resolution = 224
#     resolution = resolution
#
#     input_data = torch.randn((bs, 3, resolution, resolution), device='cuda').cuda()
#
#     if channel_last:
#         input_data = input_data.to(memory_format=torch.channels_last)
#         model = model.to(memory_format=torch.channels_last)
#
#     model.cuda()
#     model.eval()
#
#
#     with torch.cuda.amp.autocast(dtype=torch.bfloat16):
#         output_bfloat16 = model(input_data)
#
#
#     if args.trt:
#         model.cpu()
#         input_data = input_data.cpu()
#         _ = model(input_data)
#     else:
#         _ = model(input_data)
#
#     if 1:
#         for module in model.modules():
#             if hasattr(module, 'switch_to_deploy'):
#                 module.switch_to_deploy()
#
#     if args.trt:
#         from models.latencyestimator import compute_latency_trt
#
#
#
#         if args.sparse:
#             model = AmpereModule(model)
#             model.create_ampere_mask()
#
#         # print(args.sparse)
#         time_per_op = compute_latency_trt(model, 3, resolution, resolution, bs, enable_fp16=True,
#                                           sparsity=args.sparse)
#
#         print(f"TRT latency: {time_per_op}")
#         print(f"Throughput med {1.0 / (time_per_op)}")
#
#     output = model(input_data)
#
#
#
#     # train check
#     loss = output.pow(2).sum()
#     loss.backward()
#     print(output.shape)
#
#     model.eval()
#
#
#
#     macs, params = get_model_complexity_info(model, tuple([3, resolution, resolution]),
#                                              as_strings=False, print_per_layer_stat=False, verbose=False)
#
#     print(f"Model stats: macs: {macs}, and params: {params}")
#
#
#
#
#     if compute_latency:
#         from models.latencyestimator import compute_latency_trt
#         compute_latency_trt(model, 3, 224, 224, bs, enable_fp16=True)
#
#     with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
#         with record_function("model_inference"):
#             with torch.cuda.amp.autocast():
#                 output = model(input_data)
#
#     prof.export_chrome_trace("trace.json")
#     print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
#
#     # warm up
#     with torch.cuda.amp.autocast():
#         for ii in range(100):
#             output = model(input_data)
#
#     # speed
#     import time
#     import numpy as np
#
#     timer = []
#     start_time = time.time()
#     with torch.cuda.amp.autocast(True):
#
#         for ii in range(300):
#             start_time_loc = time.time()
#
#             output = model(input_data)
#
#             timer.append(time.time()-start_time_loc)
#         torch.cuda.synchronize()
#     end_time = time.time()
#     print(f"Throughput {bs * 1.0 / ((end_time - start_time) / 300)}")
#     print(f"Throughput Med {int(bs * 1.0 / ((np.median(timer))))}")
#
#
#     # warm up
#     with torch.cuda.amp.autocast():
#         output_float16 = model(input_data)
#     with torch.cuda.amp.autocast(dtype=torch.bfloat16):
#         output_bfloat16 = model(input_data)
#     with torch.cuda.amp.autocast(False):
#         output_float32 = model(input_data)
#
#     print("Float16-bfloat16", (output_float16-output_bfloat16).pow(2).mean())
#     print("Float32-bfloat16", (output_float32-output_bfloat16).pow(2).mean())
#     print("Float32-float16", (output_float32-output_float16).pow(2).mean())