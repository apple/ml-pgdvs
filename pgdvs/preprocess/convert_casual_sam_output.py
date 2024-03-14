import os
import sys
import json
import tqdm
import pathlib
import argparse
import PIL.Image
import numpy as np


def extract_depth(base_dir, rgb_dir, save_dir):
    result_dir = base_dir / "BA_full"
    result_f_list = sorted(list(result_dir.glob("*.npz")))

    img_exts = PIL.Image.registered_extensions()
    supported_img_exts = {ex for ex, f in img_exts.items() if f in PIL.Image.OPEN}
    img_f_list = []
    for tmp_ext in supported_img_exts:
        img_f_list = img_f_list + list(rgb_dir.glob(f"*{tmp_ext}"))

    img_f_list = sorted(img_f_list)

    assert len(img_f_list) == len(
        result_f_list
    ), f"{len(img_f_list)}, {len(result_f_list)}"

    K0 = None

    pose_dir = save_dir / "poses"
    pose_dir.mkdir(exist_ok=True, parents=True)

    depth_dir = save_dir / "depths"
    depth_dir.mkdir(exist_ok=True, parents=True)

    for tmp_i, tmp_result_f in enumerate(tqdm.tqdm(result_f_list)):
        tmp_idx = int(tmp_result_f.stem)  # 0002.npz
        assert tmp_idx == tmp_i, f"{tmp_idx}, {tmp_i}"

        # disp (192, 384) float32
        # R (3, 3) float32
        # t (3,) float32
        # K (3, 3) float32
        # img (192, 384, 3) float32
        # mask_motion (192, 384) float32
        # uncertainty_pred (2, 192, 384) float32
        # gt_K (3, 3) float32
        tmp_info = np.load(tmp_result_f)
        disp = tmp_info["disp"]
        depth = 1 / (disp + 1e-8)
        cam_c2w = np.eye(4)
        cam_c2w[:3, :3] = tmp_info["R"]
        cam_c2w[:3, 3] = tmp_info["t"]
        K = np.eye(4)
        K[:3, :3] = tmp_info["K"]  # this is important

        if K0 is None:
            K0 = K
        else:
            assert np.sum(np.abs(K0 - K)) < 1e-5, f"{K0}, {K}"

        pose_f = pose_dir / f"{img_f_list[tmp_i].stem}.npz"
        pose_dict = {"c2w": cam_c2w, "K": K}
        np.savez(pose_f, **pose_dict)

        depth_f = depth_dir / f"{img_f_list[tmp_i].stem}.npz"
        np.savez(depth_f, depth=depth)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--casual_sam_dir",
        type=str,
        help="directory for results from CasualSAM",
    )
    parser.add_argument("--rgb_dir", type=str, help="rgb directory")
    parser.add_argument("--save_dir", type=str, help="save directory")

    args = parser.parse_args()

    casual_sam_dir = pathlib.Path(args.casual_sam_dir)
    rgb_dir = pathlib.Path(args.rgb_dir)
    save_dir = pathlib.Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    extract_depth(casual_sam_dir, rgb_dir, save_dir)
    print("Done with extracing poses from CasualSAM")
