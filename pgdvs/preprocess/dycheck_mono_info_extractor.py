import os
import sys
import cv2
import math
import tqdm
import shutil
import pathlib
import argparse
import PIL.Image
import numpy as np

from pgdvs.datasets.dycheck_utils import iPhoneParser


def extract_frame_info(parser, time_id, camera_id):
    rgba = parser.load_rgba(time_id, camera_id)
    # rgb = rgba.astype(np.float32)[..., :3] / 255.0
    rgb = rgba[..., :3]

    depth = parser.load_depth(time_id, camera_id)[..., 0]  # [H, W, 1] -> [H, W]

    cam = parser.load_camera(time_id, camera_id)  # [H, W, 1]
    K = cam.intrin  # [3, 3]
    w2c = cam.extrin  # [4, 4]

    return rgb, depth, K, w2c


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=".")
    parser.add_argument("--save_dir", type=str, default=".")
    parser.add_argument("--scene_id", type=str, default="apple")

    args = parser.parse_args()

    iphone_parser = iPhoneParser(args.scene_id, data_root=args.data_dir)

    train_frame_names, train_time_ids, train_camera_ids = iphone_parser.load_split(
        "train"
    )

    n_train_frame_names = len(train_frame_names)
    n_train_time_ids = len(train_time_ids)
    n_train_camera_ids = len(train_camera_ids)

    assert (
        n_train_frame_names == n_train_time_ids
    ), f"{n_train_frame_names}, {n_train_time_ids}"
    assert (
        n_train_frame_names == n_train_camera_ids
    ), f"{n_train_frame_names}, {n_train_camera_ids}"

    save_dir = pathlib.Path(args.save_dir) / args.scene_id
    save_dir.mkdir(parents=True, exist_ok=True)

    rgb_dir = save_dir / "rgbs"
    rgb_dir.mkdir(parents=True, exist_ok=True)

    depth_dir = save_dir / "depths"
    depth_dir.mkdir(parents=True, exist_ok=True)

    all_K = []
    all_w2c = []

    for i in tqdm.tqdm(range(n_train_camera_ids), desc="#train_frames converting"):
        tmp_time_id = train_time_ids[i]
        tmp_cam_id = train_camera_ids[i]
        tmp_frame_name = iphone_parser.get_frame_name(tmp_time_id, tmp_cam_id)

        tmp_rgb, tmp_depth, tmp_K, tmp_w2c = extract_frame_info(
            iphone_parser, tmp_time_id, tmp_cam_id
        )

        PIL.Image.fromarray(tmp_rgb).save(rgb_dir / f"{tmp_frame_name}.png")

        with open(depth_dir / f"{tmp_frame_name}.npy", "wb") as f:
            np.save(f, tmp_depth)

        all_K.append(tmp_K)
        all_w2c.append(tmp_w2c)

    all_K = np.array(all_K)  # [#frame, 3, 3]
    all_w2c = np.array(all_w2c)  # [#frame, 4, 4]

    print("\nall_K: ", all_K.shape, all_w2c.shape, "\n")
    np.savez(save_dir / "camera.npz", all_K=all_K, all_w2c=all_w2c)
