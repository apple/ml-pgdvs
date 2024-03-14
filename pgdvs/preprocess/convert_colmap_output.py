# Modified from https://github.com/zhengqili/Neural-Scene-Flow-Fields/blob/d4001759a39b056c95d8bc22da34b10b4fb85afb/nsff_scripts/save_poses_nerf.py

import os
import sys
import json
import tqdm
import pathlib
import argparse
import numpy as np

import pgdvs.preprocess.colmap_reader as colmap_reader


def hwf_to_K(hwf):
    h, w, f = hwf[:, 0].tolist()  # [3, 1] -> [3,]
    print(h, w, f)
    K = np.eye(3)
    K[0, 0] = f
    K[1, 1] = f
    K[0, 2] = w / 2.0
    K[1, 2] = h / 2.0

    return K


def get_bbox_corners(points):
    lower = points.min(axis=0)
    upper = points.max(axis=0)
    return np.stack([lower, upper])


def filter_outlier_points(points, inner_percentile):
    """Filters outlier points."""
    outer = 1.0 - inner_percentile
    lower = outer / 2.0
    upper = 1.0 - lower
    centers_min = np.quantile(points, lower, axis=0)
    centers_max = np.quantile(points, upper, axis=0)
    result = points.copy()

    too_near = np.any(result < centers_min[None, :], axis=1)
    too_far = np.any(result > centers_max[None, :], axis=1)

    return result[~(too_near | too_far)]


def load_colmap_data(realdir, save_dir):
    camerasfile = os.path.join(realdir, "sparse/cameras.bin")
    camdata = colmap_reader.read_cameras_binary(camerasfile)

    list_of_keys = list(camdata.keys())
    assert (
        len(list_of_keys) == 1
    ), f"{list_of_keys}"  # check that there is only one track
    cam = camdata[list_of_keys[0]]
    print("Cameras", len(cam))

    h, w, f = cam.height, cam.width, cam.params[0]
    # w, h, f = factor * w, factor * h, factor * f
    hwf = np.array([h, w, f]).reshape([3, 1])

    imagesfile = os.path.join(realdir, "sparse/images.bin")
    imdata = colmap_reader.read_images_binary(imagesfile)

    w2c_mats = []
    bottom = np.array([0, 0, 0, 1.0]).reshape([1, 4])

    names = [imdata[k].name for k in imdata]
    img_keys = [k for k in imdata]

    print("Images #", len(names))
    perm = np.argsort(names)

    points3dfile = os.path.join(realdir, "sparse/points3D.bin")
    pts3d = colmap_reader.read_points3d_binary(points3dfile)

    # extract point 3D xyz
    point_cloud = []
    for key in pts3d:
        point_cloud.append(pts3d[key].xyz)

    point_cloud = np.stack(point_cloud, 0)
    point_cloud = filter_outlier_points(point_cloud, 0.95)

    bounds_mats = []

    upper_bound = 1000

    if upper_bound < len(img_keys):
        print("Only keeping " + str(upper_bound) + " images!")

    for i in tqdm.tqdm(perm[0 : min(upper_bound, len(img_keys))], desc="#images"):
        im = imdata[img_keys[i]]
        # print(im.name)
        R = im.qvec2rotmat()
        t = im.tvec.reshape([3, 1])
        m = np.concatenate([np.concatenate([R, t], 1), bottom], 0)
        w2c_mats.append(m)

        pts_3d_idx = im.point3D_ids
        pts_3d_vis_idx = pts_3d_idx[pts_3d_idx >= 0]

        #
        depth_list = []
        for k in range(len(pts_3d_vis_idx)):
            point_info = pts3d[pts_3d_vis_idx[k]]
            P_g = point_info.xyz
            P_c = np.dot(R, P_g.reshape(3, 1)) + t.reshape(3, 1)
            depth_list.append(P_c[2])

        zs = np.array(depth_list)
        close_depth, inf_depth = np.percentile(zs, 5), np.percentile(zs, 95)
        bounds = np.array([close_depth, inf_depth])
        bounds_mats.append(bounds)

    w2c_mats = np.stack(w2c_mats, 0)
    # bounds_mats = np.stack(bounds_mats, 0)
    c2w_mats = np.linalg.inv(w2c_mats)  # [#frame, 4, 4]

    # bbox_corners = get_bbox_corners(point_cloud)
    # also add camera
    bbox_corners = get_bbox_corners(
        np.concatenate([point_cloud, c2w_mats[:, :3, 3]], axis=0)
    )

    scene_center = np.mean(bbox_corners, axis=0)
    scene_scale = 1.0 / np.sqrt(np.sum((bbox_corners[1] - bbox_corners[0]) ** 2))

    print("bbox_corners ", bbox_corners)
    print("scene_center ", scene_center, scene_scale)

    K = np.eye(4)
    K[:3, :3] = hwf_to_K(hwf)  # [4, 4]

    n_frames = c2w_mats.shape[0]

    tiled_K = np.tile(K[np.newaxis, ...], [n_frames, 1, 1])  # [#frame, 4, 4]

    save_arr = np.concatenate(
        [c2w_mats.reshape((n_frames, 16)), tiled_K.reshape((n_frames, 16))], 1
    )  # [#frame, 32]

    print(save_arr.shape)
    np.save(save_dir / "poses.npy", save_arr)

    with open(save_dir / "scene.json", "w") as f:
        json.dump(
            {
                "scale": scene_scale,
                "center": scene_center.tolist(),
                "bbox": bbox_corners.tolist(),
            },
            f,
            indent=2,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, help="COLMAP Directory")
    parser.add_argument("--save_dir", type=str, help="save Directory")

    args = parser.parse_args()

    base_dir = args.data_path
    save_dir = pathlib.Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    load_colmap_data(base_dir, save_dir)
    print("Done with imgs2poses")
