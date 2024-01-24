# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-research. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from Deformable DETR (https://github.com/fundamentalvision/Deformable-DETR)
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# ------------------------------------------------------------------------
# Modified from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
# ------------------------------------------------------------------------


"""
Backbone modules.
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
from torchvision.models._utils import IntermediateLayerGetter
from typing import Dict, List

from util.misc import NestedTensor, is_main_process
from .position_encoding import build_position_encoding
from timm.models.layers import LayerNorm2d

from timm.models import create_model
from .fastervit import build_fastervit

def _cfg(url='', **kwargs):
    return {'url': url,
            'num_classes': 1000,
            'input_size': (3, 224, 224),
            'pool_size': None,
            'crop_pct': 0.875,
            'interpolation': 'bicubic',
            'fixed_input_size': True,
            'mean': (0.485, 0.456, 0.406),
            'std': (0.229, 0.224, 0.225),
            **kwargs
            }

default_cfgs = {
    'faster_vit_0_224': _cfg(url='https://huggingface.co/ahatamiz/FasterViT/resolve/main/fastervit_0_224_1k.pth.tar',
                             crop_pct=0.875,
                             input_size=(3, 224, 224),
                             crop_mode='center'),
    'faster_vit_0_224_hat_False': _cfg(url='https://huggingface.co/ahatamiz/FasterViT/resolve/main/fastervit_0_224_1k.pth.tar',
                             crop_pct=0.875,
                             input_size=(3, 224, 224),
                             crop_mode='center'),    
    'faster_vit_1_224': _cfg(url='https://huggingface.co/ahatamiz/FasterViT/resolve/main/fastervit_1_224_1k.pth.tar',
                             crop_pct=1.0,
                             input_size=(3, 224, 224),
                             crop_mode='center'),
    'faster_vit_2_224': _cfg(url='https://huggingface.co/ahatamiz/FasterViT/resolve/main/fastervit_2_224_1k.pth.tar',
                             crop_pct=1.0,
                             input_size=(3, 224, 224),
                             crop_mode='center'),
    'faster_vit_3_224': _cfg(url='https://huggingface.co/ahatamiz/FasterViT/resolve/main/fastervit_3_224_1k.pth.tar',
                             crop_pct=1.0,
                             input_size=(3, 224, 224),
                             crop_mode='center'),
    'faster_vit_4_224': _cfg(url='https://huggingface.co/ahatamiz/FasterViT/resolve/main/fastervit_4_224_1k.pth.tar',
                             crop_pct=1.0,
                             input_size=(3, 224, 224),
                             crop_mode='center'),
    'faster_vit_5_224': _cfg(url='https://huggingface.co/ahatamiz/FasterViT/resolve/main/fastervit_5_224_1k.pth.tar',
                             crop_pct=1.0,
                             input_size=(3, 224, 224),
                             crop_mode='center'),
    'faster_vit_6_224': _cfg(url='https://huggingface.co/ahatamiz/FasterViT/resolve/main/fastervit_6_224_1k.pth.tar',
                             crop_pct=1.0,
                             input_size=(3, 224, 224),
                             crop_mode='center'),
    'faster_vit_4_21k_224': _cfg(url='https://huggingface.co/ahatamiz/FasterViT/resolve/main/fastervit_4_21k_224_w14.pth.tar',
                             crop_pct=0.95,
                             input_size=(3, 224, 224),
                             crop_mode='squash'),                          
    'faster_vit_4_21k_384': _cfg(url='https://huggingface.co/ahatamiz/FasterViT/resolve/main/fastervit_4_21k_384_w24.pth.tar',
                             crop_pct=1.0,
                             input_size=(3, 384, 384),
                             crop_mode='squash'),
    'faster_vit_4_21k_512': _cfg(url='https://huggingface.co/ahatamiz/FasterViT/resolve/main/fastervit_4_21k_512_w32.pth.tar',
                             crop_pct=1.0,
                             input_size=(3, 512, 512),
                             crop_mode='squash'),
    'faster_vit_4_21k_768': _cfg(url='https://huggingface.co/ahatamiz/FasterViT/resolve/main/fastervit_4_21k_768_w48.pth.tar',
                             crop_pct=0.93,
                             input_size=(3, 768, 768),
                             crop_mode='squash'),                                                         
}


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))
        self.eps = eps

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict,
                              missing_keys, unexpected_keys, error_msgs):
        num_batches_tracked_key = prefix + 'num_batches_tracked'
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = self.eps
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self.strides = backbone.strides
        self.num_channels = backbone.num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in sorted(xs.items()):
            out.append(x)

        # position encoding
        for x in out:
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


class Joiner_FasterViT(nn.Sequential):
    def __init__(self, backbone, position_embedding, strides, num_channels):
        super().__init__(backbone, position_embedding)
        self.strides = strides
        self.num_channels = num_channels

    def forward(self, tensor_list: NestedTensor):
        xs = self[0](tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for name, x in sorted(xs.items()):
            out.append(x)

        # position encoding
        for x in out:
            pos.append(self[1](x).to(x.tensors.dtype))

        return out, pos


def build_faster_vit_backbone(args):
    position_embedding = build_position_encoding(args)

    if args.backbone == 'faster_vit_0_224' or args.backbone == 'faster_vit_0_224_hat_False':
        num_channels=[64, 128, 256, 512]
    elif args.backbone == 'faster_vit_2_224':
        num_channels=[96, 192, 384, 768]
    elif args.backbone == 'faster_vit_4_21k_224' or args.backbone == 'faster_vit_4_21k_384' or args.backbone == 'faster_vit_4_21k_512':
        num_channels=[196, 392, 784, 1568]
    else:
        num_channels=[128, 256, 512, 512]
    
    if args.backbone in ['faster_vit_0_224',
                           'faster_vit_0_224_hat_False',
                           'faster_vit_1_224',
                           'faster_vit_2_224',
                           'faster_vit_3_224',
                           'faster_vit_4_224', 
                           'faster_vit_4_21k_224', 
                           'faster_vit_4_21k_384',
                           'faster_vit_4_21k_512']:
        return_interm_indices = [0,1,2,3]

        if args.stem_norm_type == 'BatchNorm2d':
            stem_norm_layer = nn.BatchNorm2d
        elif args.stem_norm_type == 'FrozenBatchNorm2d':
            stem_norm_layer = FrozenBatchNorm2d  
        elif args.stem_norm_type == 'LayerNorm':
            stem_norm_layer = LayerNorm2d
        else:
            raise ValueError("Invalid stem_norm_type. Allowed values are 'BatchNorm2d', 'FrozenBatchNorm2d', or 'LayerNorm2d'.")
        
        if args.output_norm_type == 'BatchNorm2d':
            output_norm_type = nn.BatchNorm2d
        elif args.output_norm_type == 'FrozenBatchNorm2d':
            output_norm_type = FrozenBatchNorm2d  
        elif args.output_norm_type == 'LayerNorm':
            output_norm_type = LayerNorm2d
        else:
            raise ValueError("Invalid output_norm_type. Allowed values are 'BatchNorm2d', 'FrozenBatchNorm2d', or 'LayerNorm2d'.")
        
        backbone = build_fastervit(args.backbone, stem_norm_layer, output_norm_type,
                               out_indices=tuple(return_interm_indices))
        
        if args.pretrained_backbone:
            import os
            from pathlib import Path

            pretrained_model_path = os.path.join(args.pretrained_dir, os.path.basename(default_cfgs[args.backbone]['url']))
            if not Path(pretrained_model_path).is_file():
                url = default_cfgs[args.backbone]['url']
                torch.hub.download_url_to_file(url=url, dst=pretrained_model_path)
            backbone._load_state_dict(pretrained_model_path)
            print("Integrated FasterViT pretrained backbone model is loaded")
        else:
            print("Scratch Integrated FasterViT model is used (Not using pretrained backbone model)")

    model = Joiner_FasterViT(backbone, position_embedding, strides=[8, 16, 32, 32], num_channels=num_channels)

    return model