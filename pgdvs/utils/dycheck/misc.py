#!/usr/bin/env python3
#
# Functions are gathered and modified from https://github.com/KAIR-BAIR/dycheck/tree/main
#
# File   : image.py
# Author : Hang Gao
# Email  : hangg.sv7@gmail.com
#
# Copyright 2022 Adobe. All rights reserved.
#
# This file is licensed to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR REPRESENTATIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

import math
from absl import logging
import itertools
from typing import Optional, Sequence, Any, Tuple
import tqdm
import cv2
import numpy as np

from pgdvs.utils.dycheck.camera import DyCheckCamera


UINT8_MAX = 255
UINT16_MAX = 65535


def to_uint8(img: np.ndarray) -> np.ndarray:
    img = np.array(img)
    if img.dtype == np.uint8:
        return img
    if not issubclass(img.dtype.type, np.floating):
        raise ValueError(
            f"Input image should be a floating type but is of type " f"{img.dtype!r}."
        )
    return (img * UINT8_MAX).clip(0.0, UINT8_MAX).astype(np.uint8)


def to_float32(img: np.ndarray) -> np.ndarray:
    img = np.array(img)
    if img.dtype == np.float32:
        return img

    dtype = img.dtype
    img = img.astype(np.float32)
    if dtype == np.uint8:
        return img / UINT8_MAX
    elif dtype == np.uint16:
        return img / UINT16_MAX
    elif dtype == np.float64:
        return img
    elif dtype == np.float16:
        return img

    raise ValueError(f"Unexpected dtype: {dtype}.")


def downscale(img: np.ndarray, scale: int) -> np.ndarray:
    if scale == 1:
        return img

    height, width = img.shape[:2]
    if height % scale > 0 or width % scale > 0:
        raise ValueError(
            f"Image shape ({height},{width}) must be divisible by the"
            f" scale ({scale})."
        )
    out_height, out_width = height // scale, width // scale
    resized = cv2.resize(img, (out_width, out_height), cv2.INTER_AREA)
    return resized


def upscale(img: np.ndarray, scale: int) -> np.ndarray:
    if scale == 1:
        return img

    height, width = img.shape[:2]
    out_height, out_width = height * scale, width * scale
    resized = cv2.resize(img, (out_width, out_height), cv2.INTER_AREA)
    return resized


def rescale(
    img: np.ndarray, scale_factor: float, interpolation: Any = cv2.INTER_AREA
) -> np.ndarray:
    scale_factor = float(scale_factor)

    if scale_factor <= 0.0:
        raise ValueError("scale_factor must be a non-negative number.")
    if scale_factor == 1.0:
        return img

    height, width = img.shape[:2]
    if scale_factor.is_integer():
        return upscale(img, int(scale_factor))

    inv_scale = 1.0 / scale_factor
    if (
        inv_scale.is_integer()
        and (scale_factor * height).is_integer()
        and (scale_factor * width).is_integer()
    ):
        return downscale(img, int(inv_scale))

    logging.warning(
        "Resizing image by non-integer factor %f, this may lead to artifacts.",
        scale_factor,
    )

    height, width = img.shape[:2]
    out_height = math.ceil(height * scale_factor)
    out_height -= out_height % 2
    out_width = math.ceil(width * scale_factor)
    out_width -= out_width % 2

    return resize(img, (out_height, out_width), interpolation)


def resize(
    img: np.ndarray,
    shape: Tuple[int, int],
    interpolation: Any = cv2.INTER_AREA,
) -> np.ndarray:
    out_height, out_width = shape
    return cv2.resize(
        img,
        (out_width, out_height),
        interpolation=interpolation,
    )


def sobel_by_quantile(img_points: np.ndarray, q: float):
    """Return a boundary mask where 255 indicates boundaries (where gradient is
    bigger than quantile).
    """
    dx0 = np.linalg.norm(img_points[1:-1, 1:-1] - img_points[1:-1, :-2], axis=-1)
    dx1 = np.linalg.norm(img_points[1:-1, 1:-1] - img_points[1:-1, 2:], axis=-1)
    dy0 = np.linalg.norm(img_points[1:-1, 1:-1] - img_points[:-2, 1:-1], axis=-1)
    dy1 = np.linalg.norm(img_points[1:-1, 1:-1] - img_points[2:, 1:-1], axis=-1)
    dx01 = (dx0 + dx1) / 2
    dy01 = (dy0 + dy1) / 2
    dxy01 = np.linalg.norm(np.stack([dx01, dy01], axis=-1), axis=-1)

    # (H, W, 1) uint8
    boundary_mask = (dxy01 > np.quantile(dxy01, q)).astype(np.float32)
    boundary_mask = (
        np.pad(boundary_mask, ((1, 1), (1, 1)), constant_values=False)[
            ..., None
        ].astype(np.uint8)
        * 255
    )
    return boundary_mask


def dilate(img: np.ndarray, kernel_size: Optional[int]):
    if kernel_size is None:
        return img
    is_float = np.issubdtype(img.dtype, np.floating)
    img = to_uint8(img)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    dilated = cv2.dilate(img, kernel, iterations=1)
    if is_float:
        dilated = to_float32(dilated)
    return dilated


def tsdf_fusion(
    imgs: np.ndarray,
    depths: np.ndarray,
    cameras: Sequence[DyCheckCamera],
    *,
    voxel_length: float = 1,
    sdf_trunc: float = 0.01,
    depth_far: float = 1e5,
):
    import open3d as o3d

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    for rgb, depth, camera in zip(
        tqdm.tqdm(imgs, desc="* Fusing RGBDs"),
        depths,
        cameras,
    ):
        if (depth != 0).sum() == 0:
            continue
        # Make sure that the RGBD image is contiguous.
        rgb = o3d.geometry.Image(np.array(rgb))
        depth = o3d.geometry.Image(np.array(depth))
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            rgb,
            depth,
            depth_scale=1,
            depth_trunc=depth_far,
            convert_rgb_to_intensity=False,
        )
        w2c = camera.extrin
        W, H = camera.image_size
        fx = fy = camera.focal_length
        cx, cy = camera.principal_point
        volume.integrate(
            rgbd, o3d.camera.PinholeCameraIntrinsic(W, H, fx, fy, cx, cy), w2c
        )

    pcd = volume.extract_point_cloud()
    return np.asarray(pcd.points), np.asarray(pcd.colors)


def get_bbox_segments(bbox: np.ndarray):
    points = x000, x001, x010, x011, x100, x101, x110, x111 = np.array(
        list(itertools.product(*bbox.T.tolist()))
    )
    end_points = [x001, x011, x000, x010, x101, x111, x100, x110]
    points = points.tolist()
    points += [x000, x001, x010, x011]
    end_points += [x100, x101, x110, x111]

    return np.array(points), np.array(end_points)


def tringulate_rays(origins: np.ndarray, viewdirs: np.ndarray) -> np.ndarray:
    """Triangulate a set of rays to find a single lookat point.

    Args:
        origins (np.ndarray): A (N, 3) array of ray origins.
        viewdirs (np.ndarray): A (N, 3) array of ray view directions.

    Returns:
        np.ndarray: A (3,) lookat point.
    """
    import tensorflow as tf
    from tensorflow_graphics.geometry.representation.ray import (
        triangulate as ray_triangulate,
    )

    tf.config.set_visible_devices([], "GPU")

    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

    origins = np.array(origins[None], np.float32)
    viewdirs = np.array(viewdirs[None], np.float32)
    weights = np.ones(origins.shape[:2], dtype=np.float32)
    points = np.array(ray_triangulate(origins, origins + viewdirs, weights))
    return points[0]
