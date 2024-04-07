# Copyright (C) 2024-present Naver Corporation. All rights reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
#
# --------------------------------------------------------
# utilities for interpreting the DUST3R output
# --------------------------------------------------------
import numpy as np
import torch
from dust3r.utils.geometry import xy_grid


def estimate_focal_knowing_depth(pts3d, pp, focal_mode='median', min_focal=0.5, max_focal=3.5):
    """절대 깊이가 알려진 경우 재투영 방법을 사용하여 카메라 초점을 추정합니다:
        1) 강건한 추정기를 사용하여 카메라 초점을 추정합니다.
        2) 특정 오차를 최소화하도록 실제 광선에 점을 재투영합니다.
    """
    B, H, W, THREE = pts3d.shape
    assert THREE == 3

    # 중심 픽셀 그리드
    pixels = xy_grid(W, H, device=pts3d.device).view(1, -1, 2) - pp.view(-1, 1, 2)  # B,HW,2
    pts3d = pts3d.flatten(1, 2)  # (B, HW, 3)

    if focal_mode == 'median':
        with torch.no_grad():
            # 초점의 직접 추정
            u, v = pixels.unbind(dim=-1)
            x, y, z = pts3d.unbind(dim=-1)
            fx_votes = (u * z) / x
            fy_votes = (v * z) / y

            # 정사각형 픽셀을 가정하므로 X와 Y에 대해 동일한 초점을 가짐
            f_votes = torch.cat((fx_votes.view(B, -1), fy_votes.view(B, -1)), dim=-1)
            focal = torch.nanmedian(f_votes, dim=-1).values

    elif focal_mode == 'weiszfeld':
        # l2 폐형으로 초점 초기화
        # 초점 = argmin Sum | pixel - 초점 * (x,y)/z|를 찾으려고 함
        xy_over_z = (pts3d[..., :2] / pts3d[..., 2:3]).nan_to_num(posinf=0, neginf=0)  # 동차 좌표 (x,y,1)

        dot_xy_px = (xy_over_z * pixels).sum(dim=-1)
        dot_xy_xy = xy_over_z.square().sum(dim=-1)

        focal = dot_xy_px.mean(dim=1) / dot_xy_xy.mean(dim=1)

        # 반복적인 가중치 최소제곱법
        for iter in range(10):
            # 거리의 역수로 가중치 조정
            dis = (pixels - focal.view(-1, 1, 1) * xy_over_z).norm(dim=-1)
            # print(dis.nanmean(-1))
            w = dis.clip(min=1e-8).reciprocal()
            # 새로운 가중치로 스케일링 업데이트
            focal = (w * dot_xy_px).mean(dim=1) / (w * dot_xy_xy).mean(dim=1)
    else:
        raise ValueError(f'잘못된 {focal_mode=} 값입니다.')

    focal_base = max(H, W) / (2 * np.tan(np.deg2rad(60) / 2))  # size / 1.1547005383792515
    focal = focal.clip(min=min_focal*focal_base, max=max_focal*focal_base)
    # print(focal)
    return focal
