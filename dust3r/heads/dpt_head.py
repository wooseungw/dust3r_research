# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# dpt head implementation for DUST3R
# Downstream heads assume inputs of size B x N x C (where N is the number of tokens) ;
# or if it takes as input the output at every layer, the attribute return_all_layers should be set to True
# the forward function also takes as input a dictionnary img_info with key "height" and "width"
# for PixelwiseTask, the output will be of dimension B x num_channels x H x W
# --------------------------------------------------------
from einops import rearrange
from typing import List
import torch
import torch.nn as nn
from dust3r.heads.postprocess import postprocess
import dust3r.utils.path_to_croco  # noqa: F401
from models.dpt_block import DPTOutputAdapter  # noqa


class DPTOutputAdapter_fix(DPTOutputAdapter):

    """
    dust3r를 위해 croco의 DPTOutputAdapter 구현을 수정:
    중복된 가중치를 제거하고 dust3r에 맞게 forward 함수를 수정합니다.
    """

    def init(self, dim_tokens_enc=768):
        super().init(dim_tokens_enc)
        # 중복된 가중치들을 삭제합니다.
        del self.act_1_postprocess
        del self.act_2_postprocess
        del self.act_3_postprocess
        del self.act_4_postprocess

    def forward(self, encoder_tokens: List[torch.Tensor], image_size=None):
        assert self.dim_tokens_enc is not None, 'init(dim_tokens_enc) 함수를 먼저 호출해야 합니다.'
        # H, W = input_info['image_size']
        image_size = self.image_size if image_size is None else image_size
        H, W = image_size
        # 높이와 너비에 대한 패치 수
        N_H = H // (self.stride_level * self.P_H)
        N_W = W // (self.stride_level * self.P_W)

        # 지정된 ViT 레이어에서 4개의 레이어에 디코더를 연결합니다.
        layers = [encoder_tokens[hook] for hook in self.hooks]

        # 전역 토큰을 무시하고 작업에 필요한 토큰만 추출합니다.
        layers = [self.adapt_tokens(l) for l in layers]

        # 토큰을 공간적인 표현으로 변환합니다.
        layers = [rearrange(l, 'b (nh nw) c -> b c nh nw', nh=N_H, nw=N_W) for l in layers]

        layers = [self.act_postprocess[idx](l) for idx, l in enumerate(layers)]
        # 선택한 특징 차원으로 레이어를 투영합니다.
        layers = [self.scratch.layer_rn[idx](l) for idx, l in enumerate(layers)]

        # 개선 단계를 사용하여 레이어를 퓨즈합니다.
        path_4 = self.scratch.refinenet4(layers[3])[:, :, :layers[2].shape[2], :layers[2].shape[3]]
        path_3 = self.scratch.refinenet3(path_4, layers[2])
        path_2 = self.scratch.refinenet2(path_3, layers[1])
        path_1 = self.scratch.refinenet1(path_2, layers[0])

        # 출력 헤드
        out = self.head(path_1)

        return out


class PixelwiseTaskWithDPT(nn.Module):
    """ dust3r를 위한 DPT 모듈, 모든 픽셀에 대한 3D 포인트와 신뢰도를 반환할 수 있습니다. """


    def __init__(self, *, n_cls_token=0, hooks_idx=None, dim_tokens=None,
                 output_width_ratio=1, num_channels=1, postprocess=None, depth_mode=None, conf_mode=None, **kwargs):
        super(PixelwiseTaskWithDPT, self).__init__()
        self.return_all_layers = True  # backbone needs to return all layers
        self.postprocess = postprocess
        self.depth_mode = depth_mode
        self.conf_mode = conf_mode

        assert n_cls_token == 0, "Not implemented"
        dpt_args = dict(output_width_ratio=output_width_ratio,
                        num_channels=num_channels,
                        **kwargs)
        if hooks_idx is not None:
            dpt_args.update(hooks=hooks_idx)
        self.dpt = DPTOutputAdapter_fix(**dpt_args)
        dpt_init_args = {} if dim_tokens is None else {'dim_tokens_enc': dim_tokens}
        self.dpt.init(**dpt_init_args)

    def forward(self, x, img_info):
        out = self.dpt(x, image_size=(img_info[0], img_info[1]))
        if self.postprocess:
            out = self.postprocess(out, self.depth_mode, self.conf_mode)
        return out


def create_dpt_head(net, has_conf=False):
    """
    주어진 net 매개변수에 대한 PixelwiseTaskWithDPT를 반환합니다.

    Parameters:
        net (object): 네트워크 객체입니다.
        has_conf (bool, optional): Confidence 값을 가지는지 여부를 나타내는 불리언 값입니다. 기본값은 False입니다.

    Returns:
        object: PixelwiseTaskWithDPT 객체를 반환합니다.
    """
    # assert 문은 주어진 조건이 참이 아닐 경우 프로그램을 중단시키는 역할을 합니다.
    assert net.dec_depth > 9
    l2 = net.dec_depth
    feature_dim = 256
    last_dim = feature_dim//2
    out_nchan = 3
    ed = net.enc_embed_dim
    dd = net.dec_embed_dim
    return PixelwiseTaskWithDPT(num_channels=out_nchan + has_conf,
                                feature_dim=feature_dim,
                                last_dim=last_dim,
                                hooks_idx=[0, l2*2//4, l2*3//4, l2],
                                dim_tokens=[ed, dd, dd, dd],
                                postprocess=postprocess,
                                depth_mode=net.depth_mode,
                                conf_mode=net.conf_mode,
                                head_type='regression')

if __name__ == '__main__':
    # dpthead의 매개변수 수 계산
    dpthead = create_dpt_head(net)
    num_params = sum(p.numel() for p in dpthead.parameters())
    print(f"dpthead의 매개변수 수: {num_params}")

    # 추론 시간을 위한 FLOPS 계산
    input_size = (1, 3, 256, 256)  # 예시 입력 크기
    flops = torch.cuda.FloatTensor(input_size).to(dpthead.device)
    flops = dpthead.forward_flops(flops)
    print(f"추론 시간을 위한 FLOPS: {flops}")
