import numpy as np


def read_poses_nvidia_long(cam_f, n_img_fs):
    poses_arr = np.load(cam_f, allow_pickle=True)  # [#frames, 17]

    # NOTE: for DynIBaR, poses are repeated. I.e., the first 12 cameras are repeated for the whole video.
    # Namely, for any i \in [0, 12], i and i + 12 x n has the same camera matrices for any n.

    assert poses_arr.shape[0] == n_img_fs, f"{poses_arr.shape}, {n_img_fs}"

    poses = poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])  # [3, 5, #frame]
    bds = poses_arr[:, -2:].transpose([1, 0])  # [2, #frame]

    # Correct rotation matrix ordering and move variable dim to axis 0
    # https://github.com/Fyusion/LLFF/blob/c6e27b1ee59cb18f054ccb0f87a90214dbe70482/README.md#using-your-own-poses-without-running-colmap
    poses = np.concatenate(
        [poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1
    )  # [down, right, back] -> [+X right, +Y up, +Z back]
    poses = np.moveaxis(poses, -1, 0).astype(
        np.float32
    )  # [3, 5, #frame] -> [#frame, 3, 5]

    # images = np.moveaxis(imgs, -1, 0).astype(np.float32)  # [#frame, H, W, 3]
    # bds = np.moveaxis(bds, -1, 0).astype(np.float32)  # [#frame, 2]

    all_hwf = poses[:, :, 4]  # [#frame, 3]
    homo_placeholder = np.zeros((n_img_fs, 1, 4))
    homo_placeholder[..., 3] = 1
    all_c2w = np.concatenate(
        (poses[:, :, :4], homo_placeholder), axis=1
    )  # [#frames, 3, 4] -> [#frames, 4, 4]

    # [right, up, back] (LLFF) -> [right, down, forward] (OpenCV's convention)
    all_c2w[..., 1:3] *= -1

    return all_c2w, all_hwf


def hwf_to_K(hwf, normalized=False, tgt_shape=None):
    h, w, f = hwf
    K = np.eye(3)
    K[0, 0] = f
    K[1, 1] = f
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0

    if tgt_shape is not None:
        tgt_h, tgt_w = tgt_shape
        K[0, :] = K[0, :] * tgt_w / w
        K[1, :] = K[1, :] * tgt_h / h
        h, w = tgt_h, tgt_w

    if normalized:
        K[0, :] = K[0, :] / w
        K[1, :] = K[1, :] / h
    return K


import argparse
import os
import cv2
import glob
import numpy as np
import torch
import pathlib
import tqdm
from PIL import Image

import torch
import torch.nn.functional as F


# ----------------------------------------------------------------------------------------------
# Flow visualization code used from https://github.com/tomrunia/OpticalFlow_Visualization


# MIT License
#
# Copyright (c) 2018 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Tom Runia
# Date Created: 2018-08-03


def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY) / RY)
    col = col + RY
    # YG
    colorwheel[col : col + YG, 0] = 255 - np.floor(255 * np.arange(0, YG) / YG)
    colorwheel[col : col + YG, 1] = 255
    col = col + YG
    # GC
    colorwheel[col : col + GC, 1] = 255
    colorwheel[col : col + GC, 2] = np.floor(255 * np.arange(0, GC) / GC)
    col = col + GC
    # CB
    colorwheel[col : col + CB, 1] = 255 - np.floor(255 * np.arange(CB) / CB)
    colorwheel[col : col + CB, 2] = 255
    col = col + CB
    # BM
    colorwheel[col : col + BM, 2] = 255
    colorwheel[col : col + BM, 0] = np.floor(255 * np.arange(0, BM) / BM)
    col = col + BM
    # MR
    colorwheel[col : col + MR, 2] = 255 - np.floor(255 * np.arange(MR) / MR)
    colorwheel[col : col + MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u) / np.pi
    fk = (a + 1) / 2 * (ncols - 1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1 - f) * col0 + f * col1
        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])
        col[~idx] = col[~idx] * 0.75  # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2 - i if convert_to_bgr else i
        flow_image[:, :, ch_idx] = np.floor(255 * col)
    return flow_image


def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, "input flow must have three dimensions"
    assert flow_uv.shape[2] == 2, "input flow must have shape [H,W,2]"
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:, :, 0]
    v = flow_uv[:, :, 1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)


# ----------------------------------------------------------------------------------------------


def bilinear_sampler(img, coords, mode="bilinear", mask=False):
    """Wrapper for grid_sample, uses pixel coordinates"""
    H, W = img.shape[-2:]
    xgrid, ygrid = coords.split([1, 1], dim=-1)
    xgrid = 2 * xgrid / (W - 1) - 1
    ygrid = 2 * ygrid / (H - 1) - 1

    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def coords_grid(batch, ht, wd, device):
    coords = torch.meshgrid(
        torch.arange(ht, device=device), torch.arange(wd, device=device)
    )
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def viz(*, img1, img2, flo12, flo21, img12, img21, occ1, occ2, save_f):
    img1 = img1[0].permute(1, 2, 0).cpu().numpy()
    img2 = img2[0].permute(1, 2, 0).cpu().numpy()
    flo12 = flo12[0].permute(1, 2, 0).cpu().numpy()  # [H, W, 2], float32
    flo21 = flo21[0].permute(1, 2, 0).cpu().numpy()

    img12 = img12[0].permute(1, 2, 0).cpu().numpy()
    img21 = img21[0].permute(1, 2, 0).cpu().numpy()

    img1_diff = np.abs(img1 - img21)
    img1_diff = img1_diff / (np.max(img1_diff) + 1e-8) * 255
    img2_diff = np.abs(img2 - img12)
    img2_diff = img2_diff / (np.max(img2_diff) + 1e-8) * 255

    occ1_rgb = img1 * occ1[..., None].astype(np.float32)
    occ1 = np.tile(occ1[..., None], [1, 1, 3]) * 255

    occ2_rgb = img2 * occ2[..., None].astype(np.float32)
    occ2 = np.tile(occ2[..., None], [1, 1, 3]) * 255

    # map flow to rgb image
    img_flo12 = flow_to_image(flo12)
    img_flo21 = flow_to_image(flo21)

    cat_img1 = np.concatenate(
        [img1, img_flo12, img21, img1_diff, occ1, occ1_rgb], axis=1
    )
    cat_img2 = np.concatenate(
        [img2, img_flo21, img12, img2_diff, occ2, occ2_rgb], axis=1
    )
    cat_img = np.concatenate([cat_img1, cat_img2], axis=0).astype(np.uint8)

    Image.fromarray(cat_img).save(save_f)

    # import matplotlib.pyplot as plt
    # plt.imshow(img_flo / 255.0)
    # plt.show()

    # cv2.imshow('image', img_flo[:, :, [2,1,0]]/255.0)
    # cv2.waitKey()

    return flo12, flo21


def warp(im2, flo12):
    """
    https://github.com/princeton-vl/RAFT/issues/64#issuecomment-748897559

    warp an image/tensor (im2) back to im1, according to the optical flow
    x: [B, C, H, W] (im2)
    flo: [B, 2, H, W] flow
    """
    B, C, H, W = im2.size()
    # mesh grid
    xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    grid = torch.cat((xx, yy), 1).float()

    if im2.is_cuda:
        grid = grid.cuda()
    vgrid = grid + flo12
    # scale grid to [-1,1]
    vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

    vgrid = vgrid.permute(0, 2, 3, 1)
    output = torch.nn.functional.grid_sample(im2, vgrid, align_corners=False)
    mask = torch.ones(im2.size()).to(vgrid.device)
    mask = torch.nn.functional.grid_sample(mask, vgrid, align_corners=False)

    mask[mask < 0.999] = 0
    mask[mask > 0] = 1

    return output


def compute_occlusion(image1, flow12, flow21, occ_thresh=1.0, return_raw=False):
    coords0 = coords_grid(1, image1.shape[2], image1.shape[3], flow12.device)
    coords1 = coords0 + flow12
    coords2 = coords1 + bilinear_sampler(flow21, coords1.permute(0, 2, 3, 1))

    err = (coords0 - coords2).norm(dim=1)

    if return_raw:
        return coords0 - coords2, err
    else:
        occ = (err[0] > occ_thresh).float().cpu().numpy()
        return occ
