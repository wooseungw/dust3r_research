# Copyright (C) 2022-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

# --------------------------------------------------------
# DPT head for ViTs
# --------------------------------------------------------
# References: 
# https://github.com/isl-org/DPT
# https://github.com/EPFL-VILAB/MultiMAE/blob/main/multimae/output_adapters.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from typing import Union, Tuple, Iterable, List, Optional, Dict

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def make_scratch(in_shape, out_shape, groups=1, expand=False):
    scratch = nn.Module()

    out_shape1 = out_shape
    out_shape2 = out_shape
    out_shape3 = out_shape
    out_shape4 = out_shape
    if expand == True:
        out_shape1 = out_shape
        out_shape2 = out_shape * 2
        out_shape3 = out_shape * 4
        out_shape4 = out_shape * 8

    scratch.layer1_rn = nn.Conv2d(
        in_shape[0],
        out_shape1,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer2_rn = nn.Conv2d(
        in_shape[1],
        out_shape2,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer3_rn = nn.Conv2d(
        in_shape[2],
        out_shape3,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )
    scratch.layer4_rn = nn.Conv2d(
        in_shape[3],
        out_shape4,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
        groups=groups,
    )

    scratch.layer_rn = nn.ModuleList([
        scratch.layer1_rn,
        scratch.layer2_rn,
        scratch.layer3_rn,
        scratch.layer4_rn,
    ])

    return scratch

class ResidualConvUnit_custom(nn.Module):
    """Residual convolution module."""

    def __init__(self, features, activation, bn):
        """Init.
        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn

        self.groups = 1

        self.conv1 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not self.bn,
            groups=self.groups,
        )

        self.conv2 = nn.Conv2d(
            features,
            features,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=not self.bn,
            groups=self.groups,
        )

        if self.bn == True:
            self.bn1 = nn.BatchNorm2d(features)
            self.bn2 = nn.BatchNorm2d(features)

        self.activation = activation

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: output
        """

        out = self.activation(x)
        out = self.conv1(out)
        if self.bn == True:
            out = self.bn1(out)

        out = self.activation(out)
        out = self.conv2(out)
        if self.bn == True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return self.skip_add.add(out, x)

class FeatureFusionBlock_custom(nn.Module):
    """Feature fusion block."""

    def __init__(
        self,
        features,
        activation,
        deconv=False,
        bn=False,
        expand=False,
        align_corners=True,
        width_ratio=1,
    ):
        """Init.
        Args:
            features (int): number of features
        """
        super(FeatureFusionBlock_custom, self).__init__()
        self.width_ratio = width_ratio

        self.deconv = deconv
        self.align_corners = align_corners

        self.groups = 1

        self.expand = expand
        out_features = features
        if self.expand == True:
            out_features = features // 2

        self.out_conv = nn.Conv2d(
            features,
            out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True,
            groups=1,
        )

        self.resConfUnit1 = ResidualConvUnit_custom(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit_custom(features, activation, bn)

        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, *xs):
        """Forward pass.
        Returns:
            tensor: output
        """
        output = xs[0]

        if len(xs) == 2:
            res = self.resConfUnit1(xs[1])
            if self.width_ratio != 1:
                res = F.interpolate(res, size=(output.shape[2], output.shape[3]), mode='bilinear')

            output = self.skip_add.add(output, res)
            # output += res

        output = self.resConfUnit2(output)

        if self.width_ratio != 1:
            # and output.shape[3] < self.width_ratio * output.shape[2]
            #size=(image.shape[])
            if (output.shape[3] / output.shape[2]) < (2 / 3) * self.width_ratio:
                shape = 3 * output.shape[3]
            else:
                shape = int(self.width_ratio * 2 * output.shape[2])
            output  = F.interpolate(output, size=(2* output.shape[2], shape), mode='bilinear')
        else:
            output = nn.functional.interpolate(output, scale_factor=2,
                    mode="bilinear", align_corners=self.align_corners)
        output = self.out_conv(output)
        return output

def make_fusion_block(features, use_bn, width_ratio=1):
    return FeatureFusionBlock_custom(
        features,
        nn.ReLU(False),
        deconv=False,
        bn=use_bn,
        expand=False,
        align_corners=True,
        width_ratio=width_ratio,
    )

class Interpolate(nn.Module):
    """Interpolation module."""

    def __init__(self, scale_factor, mode, align_corners=False):
        """Init.
        Args:
            scale_factor (float): scaling
            mode (str): interpolation mode
        """
        super(Interpolate, self).__init__()

        self.interp = nn.functional.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, x):
        """Forward pass.
        Args:
            x (tensor): input
        Returns:
            tensor: interpolated data
        """

        x = self.interp(
            x,
            scale_factor=self.scale_factor,
            mode=self.mode,
            align_corners=self.align_corners,
        )

        return x

class DsConv2d(nn.Module):
    def __init__(self, dim_in, dim_out, kernel_size, padding, stride, bias = True):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim_in, dim_in, kernel_size = kernel_size, padding = padding, groups = dim_in, stride = stride),
            nn.Conv2d(dim_in, dim_out, kernel_size = 1)
        )
    def forward(self, x):
        return self.net(x)


class DPTOutputAdapter(nn.Module):
    """DPT 출력 어댑터.

    :param num_channels: 출력 채널의 개수
    :param stride_level: 전체 이미지에 대한 스트라이드 레벨.
        예를 들어, 이미지 크기의 1/4 크기의 경우 4입니다.
    :param patch_size_full: 전체 이미지 크기에 대한 패치 크기의 정수 또는 튜플.
        작은 입력에 대한 패치 크기는 이에 따라 계산됩니다.
    :param hooks: 중간 레이어의 인덱스
    :param layer_dims: 중간 레이어의 차원
    :param feature_dim: 피처 차원
    :param last_dim: head_type이 regression인 경우 마지막 두 개의 Conv2d의 out_channels/in_channels
    :param use_bn: True로 설정하면 배치 정규화가 활성화됩니다.
    :param dim_tokens_enc: 인코더에서 나오는 토큰의 차원
    """
    def __init__(self,
                 num_channels: int = 1,
                 stride_level: int = 1,
                 patch_size: Union[int, Tuple[int, int]] = 16,
                 main_tasks: Iterable[str] = ('rgb',),
                 hooks: List[int] = [2, 5, 8, 11],
                 layer_dims: List[int] = [96, 192, 384, 768],
                 feature_dim: int = 256,
                 last_dim: int = 32,
                 use_bn: bool = False,
                 dim_tokens_enc: Optional[int] = None,
                 head_type: str = 'regression',
                 output_width_ratio=1,
                 **kwargs):
        super().__init__()
        self.num_channels = num_channels
        self.stride_level = stride_level
        self.patch_size = pair(patch_size)
        self.main_tasks = main_tasks
        self.hooks = hooks
        self.layer_dims = layer_dims
        self.feature_dim = feature_dim
        self.dim_tokens_enc = dim_tokens_enc * len(self.main_tasks) if dim_tokens_enc is not None else None
        self.head_type = head_type

        # Actual patch height and width, taking into account stride of input
        self.P_H = max(1, self.patch_size[0] // stride_level)
        self.P_W = max(1, self.patch_size[1] // stride_level)

        self.scratch = make_scratch(layer_dims, feature_dim, groups=1, expand=False)

        self.scratch.refinenet1 = make_fusion_block(feature_dim, use_bn, output_width_ratio)
        self.scratch.refinenet2 = make_fusion_block(feature_dim, use_bn, output_width_ratio)
        self.scratch.refinenet3 = make_fusion_block(feature_dim, use_bn, output_width_ratio)
        self.scratch.refinenet4 = make_fusion_block(feature_dim, use_bn, output_width_ratio)

        if self.head_type == 'regression':
            # The "DPTDepthModel" head
            self.head = nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim // 2, kernel_size=3, stride=1, padding=1),
                Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
                nn.Conv2d(feature_dim // 2, last_dim, kernel_size=3, stride=1, padding=1),
                nn.ReLU(True),
                nn.Conv2d(last_dim, self.num_channels, kernel_size=1, stride=1, padding=0)
            )
        elif self.head_type == 'semseg':
            # The "DPTSegmentationModel" head
            self.head = nn.Sequential(
                nn.Conv2d(feature_dim, feature_dim, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(feature_dim) if use_bn else nn.Identity(),
                nn.ReLU(True),
                nn.Dropout(0.1, False),
                nn.Conv2d(feature_dim, self.num_channels, kernel_size=1),
                Interpolate(scale_factor=2, mode="bilinear", align_corners=True),
            )
        else:
            raise ValueError('DPT head_type must be "regression" or "semseg".')

        if self.dim_tokens_enc is not None:
            self.init(dim_tokens_enc=dim_tokens_enc)

    def init(self, dim_tokens_enc=768):
        """
        Initialize parts of decoder that are dependent on dimension of encoder tokens.
        Should be called when setting up MultiMAE.

        :param dim_tokens_enc: Dimension of tokens coming from encoder
        """
        #print(dim_tokens_enc)

        # Set up activation postprocessing layers
        if isinstance(dim_tokens_enc, int):
            dim_tokens_enc = 4 * [dim_tokens_enc]

        self.dim_tokens_enc = [dt * len(self.main_tasks) for dt in dim_tokens_enc]
        
        self.q1 = DsConv2d(self.dim_tokens_enc[0], self.layer_dims[0], kernel_size=3, padding=1, stride=1)
        self.q2 = DsConv2d(self.dim_tokens_enc[1], self.layer_dims[1], kernel_size=3, padding=1, stride=1)
        self.q3 = DsConv2d(self.dim_tokens_enc[2], self.layer_dims[2], kernel_size=3, padding=1, stride=1)
        self.q4 = DsConv2d(self.dim_tokens_enc[3], self.layer_dims[3], kernel_size=3, padding=1, stride=1)
        
        self.k1 = DsConv2d(self.dim_tokens_enc[0], self.layer_dims[0], 3, 1, stride=2)
        self.k2 = DsConv2d(self.dim_tokens_enc[1], self.layer_dims[1], 3, 1, stride=2)
        self.k3 = DsConv2d(self.dim_tokens_enc[2], self.layer_dims[2], 3, 1, stride=2)
        self.k4 = DsConv2d(self.dim_tokens_enc[3], self.layer_dims[3], 3, 1, stride=2)
        
        self.v1 = DsConv2d(self.dim_tokens_enc[0], self.layer_dims[0], 3, 1, stride=2)
        self.v2 = DsConv2d(self.dim_tokens_enc[1], self.layer_dims[1], 3, 1, stride=2)
        self.v3 = DsConv2d(self.dim_tokens_enc[2], self.layer_dims[2], 3, 1, stride=2)
        self.v4 = DsConv2d(self.dim_tokens_enc[3], self.layer_dims[3], 3, 1, stride=2)
        
        self.act_1_postprocess = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.layer_dims[0],
                out_channels=self.layer_dims[0],
                kernel_size=4, stride=4, padding=0,
                bias=True, dilation=1, groups=1,
            )
        )

        self.act_2_postprocess = nn.Sequential(
            nn.ConvTranspose2d(
                in_channels=self.layer_dims[1],
                out_channels=self.layer_dims[1],
                kernel_size=2, stride=2, padding=0,
                bias=True, dilation=1, groups=1,
            )
        )

        self.act_3_postprocess = nn.Sequential(
            nn.Conv2d(
                in_channels=self.layer_dims[2],
                out_channels=self.layer_dims[2],
                kernel_size=1, stride=1, padding=0,
            )
        )

        self.act_4_postprocess = nn.Sequential(
            nn.Conv2d(
                in_channels=self.layer_dims[3],
                out_channels=self.layer_dims[3],
                kernel_size=3, stride=2, padding=1,
            )
        )
        self.qs = nn.ModuleList([
            self.q1,
            self.q2,
            self.q3,
            self.q4
        ])
        self.ks = nn.ModuleList([
            self.k1,
            self.k2,
            self.k3,
            self.k4
        ])
        self.vs = nn.ModuleList([
            self.v1,
            self.v2,
            self.v3,
            self.v4
        ])
        self.act_postprocess = nn.ModuleList([
            self.act_1_postprocess,
            self.act_2_postprocess,
            self.act_3_postprocess,
            self.act_4_postprocess
        ])
        
            

    def adapt_tokens(self, encoder_tokens):
        # Adapt tokens
        x = []
        x.append(encoder_tokens[:, :])
        x = torch.cat(x, dim=-1)
        return x

    def forward(self, encoder_tokens: List[torch.Tensor], image_size):
            #input_info: Dict):
        assert self.dim_tokens_enc is not None, 'Need to call init(dim_tokens_enc) function first'
        H, W = image_size
        
        # Number of patches in height and width
        N_H = H // (self.stride_level * self.P_H)
        N_W = W // (self.stride_level * self.P_W)
        
        # Hook decoder onto 4 layers from specified ViT layers
        layers = [encoder_tokens[hook] for hook in self.hooks]
        
        # Extract only task-relevant tokens and ignore global tokens.
        layers = [self.adapt_tokens(l) for l in layers]

        # Reshape tokens to spatial representation
        layers = [rearrange(l, 'b (nh nw) c -> b c nh nw', nh=N_H, nw=N_W) for l in layers]

            
        ks = [self.ks[idx](l) for idx, l in enumerate(layers)]
        print("ks",ks[0].shape)   
        print("ks",ks[1].shape)
        print("ks",ks[2].shape)
        print("ks",ks[3].shape)
        print("++++++++++++++++++++++++++++++++")
        vs = [self.vs[idx](l) for idx, l in enumerate(layers)]
        layers = [self.qs[idx](l) for idx, l in enumerate(layers)] #equal as Query
        print("layers",layers[0].shape)   
        print("layers",layers[1].shape)
        print("layers",layers[2].shape)
        print("layers",layers[3].shape)
        print("++++++++++++++++++++++++++++++++")
        
        #layers = [self.act_postprocess[idx](q,kv) for idx, q,kv in enumerate(zip(layers,kvs))]
        
        for idx,q,k,v in enumerate(zip(layers,ks,vs)):
            shape = q.shape
            q = q.flatten(2)
            k = k.flatten(2)
            v = v.flatten(2)
            attn = torch.matmul(torch.matmul(q.transpose(-1, -2), k).softmax(dim=-1), v).resize(shape)
            layers[idx] = self.act_postprocess[idx](attn)
            
        
        
        # Project layers to chosen feature dim
        layers = [self.scratch.layer_rn[idx](l) for idx, l in enumerate(layers)]

        # Fuse layers using refinement stages
        path_4 = self.scratch.refinenet4(layers[3])[:, :, :layers[2].shape[2], :layers[2].shape[3]]
        path_3 = self.scratch.refinenet3(path_4, layers[2])
        path_2 = self.scratch.refinenet2(path_3, layers[1])
        path_1 = self.scratch.refinenet1(path_2, layers[0])

        # Output head
        out = self.head(path_1)

        return out
        # Example usage of the DPTOutputAdapter model
if __name__ == '__main__':
    args = {
        'num_channels': 1,
        'stride_level': 1,
        'patch_size': 16,
        'main_tasks': ('rgb',),
        'hooks': [2, 5, 8, 11],
        'layer_dims': [96, 192, 384, 768],
        'feature_dim': 196,
        'last_dim': 32,
        'use_bn': False,
        'dim_tokens_enc': 768,
        'head_type': 'regression',
        'output_width_ratio': 1
    }
    model = DPTOutputAdapter(**args)
    model.init()
    # Create dummy input
    encoder_tokens = [(torch.ones(1,1024,768)+i) for i in range(13)]
    
    image_size = (512, 512)

    # Forward pass
    output = model(encoder_tokens, image_size)

    # Print the output
    print(output.shape)

