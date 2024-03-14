# Modified from https://github.com/deepmind/tapnet/blob/4ac6b2acd0aed36c0762f4247de9e8630340e2e0/colabs/tapir_demo.ipynb

import tree
import functools
import numpy as np
from typing import Any

import torch

import jax
import jax.dlpack
import haiku as hk
from jax.lib import xla_bridge

from pgdvs.models.tapnet import tapir_model
from pgdvs.models.tapnet.utils import transforms
from pgdvs.utils.rendering import modify_rgb_range


def jax2torch(x):
    return torch.utils.dlpack.from_dlpack(jax.dlpack.to_dlpack(x))


def torch2jax(x):
    return jax.dlpack.from_dlpack(torch.utils.dlpack.to_dlpack(x.contiguous()))


class TAPNetInterface:
    def __init__(
        self,
        ckpt_path,
        ori_rgb_range="0_1",
        query_chunk_size=4096,
        flag_keep_raw_res=False,
        local_rank=0,
    ):
        self.ori_rgb_range = ori_rgb_range
        self.query_chunk_size = query_chunk_size

        self.resize_h = 256
        self.resize_w = 256

        ckpt_state = np.load(ckpt_path, allow_pickle=True).item()
        self.params, self.state = ckpt_state["params"], ckpt_state["state"]

        build_model_func = functools.partial(
            build_model, query_chunk_size=self.query_chunk_size
        )

        model = hk.transform_with_state(build_model_func)

        if xla_bridge.get_backend().platform == "gpu":
            device = jax.devices("gpu")[local_rank]
        else:
            device = jax.devices("cpu")

        self.model_apply = jax.jit(model.apply, device=device)

        self.flag_keep_raw_res = flag_keep_raw_res

        self.network_mult = 8

    def __call__(self, *, frames, query_points):
        # frames: [N, H, W, 3]
        # query_points: [#pt, 3], 3 for [time, row, col]

        _, orig_h, orig_w, _ = frames.shape

        if self.flag_keep_raw_res:
            if (orig_h % self.network_mult != 0) or (orig_w % self.network_mult != 0):
                resize_h = int(np.ceil(orig_h / self.network_mult) * self.network_mult)
                resize_w = int(np.ceil(orig_w / self.network_mult) * self.network_mult)
            else:
                resize_h = orig_h
                resize_w = orig_w
        else:
            resize_h = self.resize_h
            resize_w = self.resize_w

        if (orig_h != resize_h) or (orig_w != resize_w):
            frames = (
                torch.nn.functional.interpolate(
                    frames.permute(0, 3, 1, 2),
                    size=(resize_h, resize_w),
                    mode="bicubic",
                    antialias=True,
                )
                .permute(0, 2, 3, 1)
                .contiguous()
            )
            query_cols_rows = transforms.convert_grid_coordinates_torch(
                query_points[:, [2, 1]],
                (orig_w, orig_h),
                (resize_w, resize_h),
            )  # transform accepts [u, v] or [col, row] format
            query_points[:, 1] = query_cols_rows[:, 1]  # for rows
            query_points[:, 2] = query_cols_rows[:, 0]  # for cols

        frames = modify_rgb_range(
            frames,
            src_range=self.ori_rgb_range,
            tgt_range="0_255",
            check_range=False,
            enforce_range=True,
        )

        frames = torch2jax(frames)
        query_points = torch2jax(query_points)

        tracks, visibles = inference(
            self.model_apply, self.params, self.state, frames, query_points
        )  # tracks: [#pt, #frame, 2], float32; visibles: [#pt, #frame], bool

        if (orig_h != resize_h) or (orig_w != resize_w):
            tracks = transforms.convert_grid_coordinates_torch(
                tracks, (resize_w, resize_h), (orig_w, orig_h)
            )

        return tracks, visibles


def build_model(frames, query_points, query_chunk_size=64):
    """Compute point tracks and occlusions given frames and query points."""
    model = tapir_model.TAPIR(
        bilinear_interp_with_depthwise_conv=False, pyramid_level=0
    )
    outputs = model(
        video=frames,
        is_training=False,
        query_points=query_points,
        query_chunk_size=query_chunk_size,
    )
    return outputs


def preprocess_frames(frames):
    """Preprocess frames to model inputs.

    Args:
      frames: [num_frames, height, width, 3], [0, 255], np.uint8

    Returns:
      frames: [num_frames, height, width, 3], [-1, 1], np.float32
    """
    frames = frames.astype(np.float32)
    frames = frames / 255 * 2 - 1
    return frames


def postprocess_occlusions(occlusions, expected_dist):
    """Postprocess occlusions to boolean visible flag.

    Args:
      occlusions: [num_points, num_frames], [-inf, inf], np.float32
      expected_dist: [num_points, num_frames], [-inf, inf], np.float32

    Returns:
      visibles: [num_points, num_frames], bool
    """
    visibles = (1 - jax.nn.sigmoid(occlusions)) * (
        1 - jax.nn.sigmoid(expected_dist)
    ) > 0.5
    return visibles


def postprocess_occlusions_torch(occlusions, expected_dist):
    """Postprocess occlusions to boolean visible flag.

    Args:
      occlusions: [num_points, num_frames], [-inf, inf], np.float32
      expected_dist: [num_points, num_frames], [-inf, inf], np.float32

    Returns:
      visibles: [num_points, num_frames], bool
    """
    visibles = (1 - torch.nn.functional.sigmoid(occlusions)) * (
        1 - torch.nn.functional.sigmoid(expected_dist)
    ) > 0.5
    return visibles


def inference(model_apply, params, state, frames, query_points):
    """Inference on one video.

    Args:
      frames: [num_frames, height, width, 3], [0, 255], np.uint8
      query_points: [num_points, 3], [0, num_frames/height/width], [t, y, x]

    Returns:
      tracks: [num_points, 3], [-1, 1], [t, y, x]
      visibles: [num_points, num_frames], bool
    """
    # Preprocess video to match model inputs format
    frames = preprocess_frames(frames)
    num_frames, height, width = frames.shape[0:3]
    query_points = query_points.astype(np.float32)
    frames, query_points = frames[None], query_points[None]  # Add batch dimension

    # Model inference
    rng = jax.random.PRNGKey(42)
    outputs, _ = model_apply(params, state, rng, frames, query_points)

    if False:
        outputs = tree.map_structure(lambda x: np.array(x[0]), outputs)
    else:
        for k in ["tracks", "occlusion", "expected_dist"]:
            outputs[k] = jax2torch(outputs[k])[0, ...]

    tracks, occlusions, expected_dist = (
        outputs["tracks"],
        outputs["occlusion"],
        outputs["expected_dist"],
    )

    # Binarize occlusions
    if False:
        visibles = postprocess_occlusions(occlusions, expected_dist)
    else:
        visibles = postprocess_occlusions_torch(occlusions, expected_dist)
    return tracks, visibles
