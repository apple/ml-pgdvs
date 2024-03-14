import os
import imageio
import tqdm
import numpy as np
from typing import List, Optional

import torch


def clamp_rgb(rgb, tgt_range="0_255"):
    assert tgt_range in ["-1_1", "0_1", "0_255"], tgt_range

    min_val = float(tgt_range.split("_")[0])
    max_val = float(tgt_range.split("_")[1])

    if isinstance(rgb, np.ndarray):
        rgb = np.clip(rgb, min_val, max_val)
    elif isinstance(rgb, torch.Tensor):
        rgb = torch.clamp(rgb, min_val, max_val)
    else:
        raise TypeError

    return rgb


def modify_rgb_range(
    rgb, src_range="0_255", tgt_range="0_255", check_range=True, enforce_range=False
):
    assert src_range in ["-1_1", "0_1", "0_255"], src_range
    assert tgt_range in ["-1_1", "0_1", "0_255"], tgt_range

    if src_range == tgt_range:
        return rgb

    if isinstance(rgb, np.ndarray):
        rgb = rgb.astype(np.float32)
    elif isinstance(rgb, torch.Tensor):
        rgb = rgb.float()
    else:
        raise TypeError

    if check_range:
        src_min = float(src_range.split("_")[0])
        src_max = float(src_range.split("_")[1])
        assert rgb.min() >= src_min, f"{rgb.min()}, {src_min}"
        assert rgb.max() <= src_max, f"{rgb.max()}, {src_max}"

    # We change ranges to be in [0, 1]
    if src_range == "0_255":
        rgb = rgb / 255.0
    elif src_range == "-1_1":
        rgb = (rgb + 1.0) / 2.0
    else:
        pass

    if enforce_range:
        if isinstance(rgb, np.ndarray):
            rgb = np.clip(rgb, 0.0, 1.0)
        elif isinstance(rgb, torch.Tensor):
            rgb = rgb.clamp(0.0, 1.0)
        else:
            raise TypeError

    # Now, RGB is in range of [0, 1]
    if tgt_range == "-1_1":
        rgb = 2.0 * rgb - 1.0
    elif tgt_range == "0_255":
        rgb = rgb * 255.0

    if check_range:
        tgt_min = float(tgt_range.split("_")[0])
        tgt_max = float(tgt_range.split("_")[1])
        assert rgb.min() >= tgt_min, f"{rgb.min()}, {tgt_min}"
        assert rgb.max() <= tgt_max, f"{rgb.max()}, {tgt_max}"

    return rgb


def images_to_video(
    images: List[np.ndarray],
    output_dir: str,
    video_name: str,
    fps: int = 10,
    quality: Optional[float] = 5,
    disable_tqdm=False,
    **kwargs,
):
    r"""Calls imageio to run FFMPEG on a list of images. For more info on
    parameters, see https://imageio.readthedocs.io/en/stable/format_ffmpeg.html
    Args:
        images: The list of images. Images should be HxWx3 in RGB order.
        output_dir: The folder to put the video in.
        video_name: The name for the video.
        fps: Frames per second for the video. Not all values work with FFMPEG,
            use at your own risk.
        quality: Default is 5. Uses variable bit rate. Highest quality is 10,
            lowest is 0.  Set to None to prevent variable bitrate flags to
            FFMPEG so you can manually specify them using output_params
            instead. Specifying a fixed bitrate using ‘bitrate’ disables
            this parameter.
    """
    assert 0 <= quality <= 10
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    video_name = video_name.replace(" ", "_").replace("\n", "_") + ".mp4"
    writer = imageio.get_writer(
        os.path.join(output_dir, video_name),
        fps=fps,
        quality=quality,
        macro_block_size=1,
        **kwargs,
    )
    # print(f"Video created: {os.path.join(output_dir, video_name)}")
    for im in tqdm.tqdm(images, disable=disable_tqdm):
        writer.append_data(im)
    writer.close()


def CreateRenderPoses(xCamTarget, N, device):
    c2w = xCamTarget.transform[0, ...]
    up = xCamTarget.transform[0, :3, 1]

    rads = torch.cat(
        (
            0.8 * torch.abs(xCamTarget.transform[0, :3, 3]),
            torch.tensor(
                [
                    1,
                ],
                device=device,
            ),
        ),
        dim=0,
    )
    rots = 2
    focal = 1
    flip = False

    def normalize(v):
        return v / torch.linalg.norm(v)

    def viewmatrix(z, up, pos):
        vec2 = normalize(z)
        vec1_avg = up
        vec0 = normalize(torch.cross(vec1_avg, vec2))
        vec1 = normalize(torch.cross(vec2, vec0))
        m = torch.stack([vec0, vec1, vec2, pos], 1)
        return m

    render_poses = []
    for theta in torch.linspace(0.0, 2.0 * np.pi * rots, N + 1)[:-1]:
        c = torch.matmul(
            c2w[:3, :4],
            torch.tensor(
                [torch.cos(theta), -torch.sin(theta), -torch.sin(theta * 0.5), 1.0],
                device=device,
            )
            * rads,
        )

        if flip:
            z = normalize(
                torch.matmul(
                    c2w[:3, :4], torch.tensor([0, 0, focal, 1.0], device=device)
                )
                - c
            )
        else:
            z = normalize(
                c
                - torch.matmul(
                    c2w[:3, :4], torch.tensor([0, 0, -focal, 1.0], device=device)
                )
            )

        render_poses.append(viewmatrix(z, up, c))
    return render_poses
