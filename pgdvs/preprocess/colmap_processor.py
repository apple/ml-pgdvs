# Modified from
# - https://github.com/kwea123/nsff_pl/blob/0e7b75543a4c3f0782332cf64c96fadce03ae34d/preprocess.py
# - https://github.com/facebookresearch/consistent_depth/blob/e2c9b724d3221aa7c0bf89aa9449ae33b418d943/tools/colmap_processor.py

import os
import pathlib
import argparse
import subprocess
import PIL.Image


def run_cmd(cmd):
    cmd_str = " ".join(cmd)
    print(f"\n{cmd_str}\n")

    new_env = os.environ.copy()
    new_env["LD_LIBRARY_PATH"] = f"/opt/conda/lib:{new_env['LD_LIBRARY_PATH']}"
    subprocess.run(cmd, env=new_env)


def run_colmap(args):
    max_num_matches = 132768  # colmap setting

    workspace_path = pathlib.Path(args.workspace_path)
    workspace_path.mkdir(parents=True, exist_ok=True)

    rgb_dir = pathlib.Path(args.image_path)
    mask_dir = pathlib.Path(args.mask_path)

    exts = PIL.Image.registered_extensions()
    supported_exts = {ex for ex, f in exts.items() if f in PIL.Image.OPEN}

    rgb_f_list = []
    for tmp_ext in supported_exts:
        rgb_f_list = rgb_f_list + list(sorted(rgb_dir.glob(f"*{tmp_ext}")))
    for tmp_rgb_f in rgb_f_list:
        tmp_mask_f = mask_dir / f"{tmp_rgb_f.name}.png"
        assert tmp_mask_f.exists(), rgb_f_list

    if not (workspace_path / "database.db").exists() or args.overwrite:
        # https://colmap.github.io/faq.html#mask-image-regions
        # Features will only be extracted from areas with mask values of 1

        # fmt: off
        cmd_feat = [
            f"{args.colmap_bin}",
            "feature_extractor",
            "--database_path", f"{args.workspace_path}/database.db",
            "--image_path", f"{args.image_path}",
            "--ImageReader.mask_path", f"{args.mask_path}",
            "--ImageReader.camera_model", "SIMPLE_RADIAL",
            "--ImageReader.single_camera", "1",
            "--SiftExtraction.num_threads", "1",
            "--SiftExtraction.gpu_index", "0",
        ]

        # "--ImageReader.default_focal_length_factor", "0.95",
        # "--SiftExtraction.peak_threshold", "0.004",
        # "--SiftExtraction.max_num_features", "8192",
        # "--SiftExtraction.edge_threshold", "16",

        # fmt: on
        run_cmd(cmd_feat)

        # fmt: off
        cmd_match = [
            f"{args.colmap_bin}",
            "exhaustive_matcher",
            "--database_path", f"{args.workspace_path}/database.db",
            "--SiftMatching.multiple_models", "1",
            "--SiftMatching.guided_matching", "1",
        ]

        # "--SiftMatching.max_ratio", "0.8",
        # "--SiftMatching.max_error", "4.0",
        # "--SiftMatching.max_distance", "0.7",
        # "--SiftMatching.max_num_matches", f"{max_num_matches}",

        # fmt: on
        run_cmd(cmd_match)

    if not (workspace_path / "sparse").exists() or args.overwrite:
        (workspace_path / "sparse").mkdir(exist_ok=True, parents=True)

        # fmt: off
        cmd_map = [
            f"{args.colmap_bin}",
            "mapper",
            "--database_path", f"{args.workspace_path}/database.db",
            "--image_path", f"{args.image_path}",
            "--output_path", f"{args.workspace_path}/sparse",
            "--Mapper.abs_pose_min_inlier_ratio", "0.5",
            "--Mapper.abs_pose_min_num_inliers", "50",
            "--Mapper.init_max_forward_motion", "1",
            "--Mapper.ba_local_num_images", "15",
        ]
        # fmt: on
        run_cmd(cmd_map)

    if not pathlib.Path(args.undistort_path).exists() or args.overwrite:
        pathlib.Path(args.undistort_path).mkdir(exist_ok=True, parents=True)

        # fmt: off
        cmd_undist = [
            f"{args.colmap_bin}",
            "image_undistorter",
            "--input_path", f"{args.workspace_path}/sparse/0",
            "--image_path", f"{args.image_path}",
            "--output_path", f"{args.undistort_path}",
            "--output_type", "COLMAP",
        ]
        # fmt: on
        run_cmd(cmd_undist)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="run colmap")
    parser.add_argument("--colmap_bin", type=str, default="colmap")
    parser.add_argument("--cuda_device", type=int, default=-1)
    parser.add_argument("--image_path", type=str)
    parser.add_argument("--mask_path", type=str)
    parser.add_argument("--workspace_path", type=str, default="colmap")
    parser.add_argument("--undistort_path", type=str, default="undistorted")
    parser.add_argument(
        "--overwrite", default=False, action="store_true", help="overwrite cache"
    )

    args = parser.parse_args()

    run_colmap(args)
