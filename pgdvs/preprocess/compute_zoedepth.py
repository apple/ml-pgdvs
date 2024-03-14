import os
import sys
import tqdm
import pathlib
import argparse
import trimesh
import PIL.Image
import numpy as np
from skimage.transform import resize as imresize
from scipy.ndimage import map_coordinates

import torch

import colmap_reader
from pgdvs.preprocess.common import read_poses_nvidia_long, hwf_to_K

cur_dir = pathlib.Path(__file__).parent.resolve()
midas_dir = cur_dir.parent.parent / "third_parties/ZoeDepth"
sys.path.append(str(midas_dir))
print("\nzoe_depth_dir: ", midas_dir, "\n")

from third_parties.ZoeDepth.zoedepth.models.builder import build_model
from third_parties.ZoeDepth.zoedepth.utils.config import get_config


TINY_VAL = 1.0e-16

FLAG_ZERO_SHIFT = False


class DepthDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root=".",
        mask_dir=".",
    ):
        super(DepthDataset, self).__init__()
        self.image_root = pathlib.Path(root)

        # print("\nself.image_root: ", self.image_root, "\n")

        exts = PIL.Image.registered_extensions()
        supported_exts = {ex for ex, f in exts.items() if f in PIL.Image.OPEN}

        image_list = []
        for tmp_ext in supported_exts:
            image_list = image_list + list(self.image_root.glob(f"*{tmp_ext}"))
        self.image_list = sorted(image_list)

        mask_list = []
        for tmp in self.image_list:
            tmp_mask_f = mask_dir / f"{int(tmp.stem):05d}_final.png"
            assert tmp_mask_f.exists(), tmp_mask_f
            mask_list.append(tmp_mask_f)
        self.mask_list = mask_list

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        return self.image_list[index], self.mask_list[index]


def get_batched_rays(*, device, batch_size, H, W, render_stride, intrinsics, c2w):
    """
    :param H: image height
    :param W: image width
    :param intrinsics: 4 by 4 intrinsic matrix
    :param c2w: 4 by 4 camera to world extrinsic matrix
    :return:
    """
    u, v = torch.meshgrid(
        torch.arange(W, device=device)[::render_stride],
        torch.arange(H, device=device)[::render_stride],
        indexing="xy",
    )  # both are [H, W]

    render_h, render_w = u.shape

    u = u.reshape(-1).float()  # + 0.5    # add half pixel
    v = v.reshape(-1).float()  # + 0.5
    pixels = torch.stack((u, v, torch.ones_like(u)), dim=0)  # (3, H*W)
    batched_pixels = pixels.unsqueeze(0).repeat(batch_size, 1, 1)  # [B, 3, HxW]

    rays_d = (
        c2w[:, :3, :3].bmm(torch.inverse(intrinsics[:, :3, :3])).bmm(batched_pixels)
    ).transpose(
        1, 2
    )  # [B, 3, 3] x [B, 3, 3] x [B, 3, HxW] -> [B, 3, HxW] -> [B, HxW, 3]
    rays_o = c2w[:, :3, 3].unsqueeze(1).repeat(1, rays_d.shape[1], 1)  # [B, HxW, 3]
    rays_d = rays_d.reshape(-1, 3)  # [BxHxW, 3]
    rays_o = rays_o.reshape(-1, 3)  # [BxHxW, 3]
    uvs = batched_pixels[:, :2, :].permute(0, 2, 1).reshape((-1, 2))

    batch_refs = (
        torch.arange(batch_size)
        .reshape((batch_size, 1))
        .expand(-1, u.shape[0])
        .reshape(-1)
    )  # [BxHxW]

    return rays_o, rays_d  # , uvs, batch_refs, (render_h, render_w)


def vis_pcl(in_disp, disp_scale, disp_shift, img, K, c2w, save_f):
    if FLAG_ZERO_SHIFT:
        disp_shift = 0.0

    tmp_disp = disp_scale * in_disp + disp_shift
    tmp_depth = 1 / (tmp_disp + TINY_VAL)

    tmp_h, tmp_w = tmp_disp.shape

    tmp_rays_o, tmp_rays_d = get_batched_rays(
        device=torch.device("cpu"),
        batch_size=1,
        H=tmp_h,
        W=tmp_w,
        render_stride=1,
        intrinsics=torch.FloatTensor(K)[None, ...],
        c2w=torch.FloatTensor(c2w)[None, ...],
    )

    tmp_pts = tmp_rays_o.numpy() + tmp_rays_d.numpy() * tmp_depth.reshape((-1, 1))

    tmp_pcl = trimesh.PointCloud(vertices=tmp_pts, colors=img.reshape(-1, 3))
    _ = tmp_pcl.export(save_f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default=".")
    parser.add_argument("--save_dir", default=".")
    parser.add_argument("--mask_dir", default=".")
    parser.add_argument("--image_subdir", default="rgbs")
    parser.add_argument(
        "--zoedepth_type", default="NK", type=str, choices=["N", "K", "NK"]
    )
    parser.add_argument("--zoedepth_ckpt_dir", type=str)
    parser.add_argument("--save_space", action="store_true")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.zoedepth_type == "N":
        ckpt_f = str(pathlib.Path(args.zoedepth_ckpt_dir) / "ZoeD_M12_N.pt")
        conf = get_config("zoedepth", "infer", pretrained_resource=f"local::{ckpt_f}")
    elif args.zoedepth_type == "K":
        ckpt_f = str(pathlib.Path(args.zoedepth_ckpt_dir) / "ZoeD_M12_K.pt")
        conf = get_config(
            "zoedepth",
            "infer",
            config_version="kitti",
            pretrained_resource=f"local::{ckpt_f}",
        )
    elif args.zoedepth_type == "NK":
        ckpt_f = str(pathlib.Path(args.zoedepth_ckpt_dir) / "ZoeD_M12_NK.pt")
        conf = get_config(
            "zoedepth_nk", "infer", pretrained_resource=f"local::{ckpt_f}"
        )
    else:
        raise ValueError(args.zoedepth_type)

    print("\nckpt_f: ", ckpt_f, "\n")
    model = build_model(conf).to(device)
    model.eval()

    root_dir = pathlib.Path(args.root_dir)

    data_dir_list = list(root_dir.glob(args.image_subdir))
    assert len(data_dir_list) == 1, f"{data_dir_list}"
    data_dir = data_dir_list[0]

    save_dir = pathlib.Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    depth_dir = save_dir / f"zoe_depths_{args.zoedepth_type.lower()}"
    depth_dir.mkdir(exist_ok=True, parents=True)

    vis_dir_share_med = depth_dir / f"vis/scale_shift_share_med"
    vis_dir_share_med.mkdir(exist_ok=True, parents=True)

    vis_dir_indiv_med = depth_dir / f"vis/scale_shift_indiv_med"
    vis_dir_indiv_med.mkdir(exist_ok=True, parents=True)

    vis_dir_share_trim = depth_dir / f"vis/scale_shift_share_trim"
    vis_dir_share_trim.mkdir(exist_ok=True, parents=True)

    vis_dir_indiv_trim = depth_dir / f"vis/scale_shift_indiv_trim"
    vis_dir_indiv_trim.mkdir(exist_ok=True, parents=True)

    mask_dir = pathlib.Path(args.mask_dir) / "masks/final"

    val_dataset = DepthDataset(root=data_dir, mask_dir=mask_dir)

    n_all_frames = len(val_dataset)
    print("\nval_dataset: ", n_all_frames, "\n")

    cam_f = root_dir / "poses_bounds_cvd.npy"
    all_c2w, all_hwf = read_poses_nvidia_long(
        cam_f, n_all_frames
    )  # c2w: [#frame, 4, 4]; hwf: [#frame, 3]

    all_w2c = np.array([np.linalg.inv(all_c2w[_, ...]) for _ in range(n_all_frames)])

    tmp_img = np.array(PIL.Image.open(val_dataset[0][0]))
    tmp_h, tmp_w, _ = tmp_img.shape
    tmp_shape = (tmp_h, tmp_w)

    # https://github.com/google/dynibar/blob/02b164144cce2d93aa4c5d87b418497286b2ae31/ibrnet/data_loaders/llff_data_utils.py#L107
    all_hwf[:, 0] = tmp_h
    all_hwf[:, 1] = tmp_w

    all_K = np.array(
        [
            hwf_to_K(all_hwf[_, :], tgt_shape=tmp_shape, normalized=False)
            for _ in range(n_all_frames)
        ]
    )  # [#frames, 3, 3]

    pts3d_f = root_dir / "sparse/points3D.bin"
    pts3d_dict = colmap_reader.read_points3d_binary(pts3d_f)

    pts3d = [pts3d_dict[_].xyz for _ in pts3d_dict]
    pts3d = np.array(pts3d).astype(np.float32)  # [#pt, 3]

    # print("pts3d: ", pts3d.shape, pts3d.dtype)

    h_pt = np.ones([pts3d.shape[0], 4])
    h_pt[:, :3] = pts3d  # [#pt, 4]
    h_pt = h_pt.T  # [4, #pt]

    print("calculating monocular depth")

    full_pred_depths = []
    pts_list = []

    mvs_depths = []
    pred_depths = []
    masks = []

    all_imgs = []
    all_pred_scale_shifts = []
    all_mvs_scale_shifts = []

    for img_i in tqdm.tqdm(range(n_all_frames), desc="#frames"):
        img_f, mask_f = val_dataset[img_i]

        img = np.asarray(PIL.Image.open(img_f)).astype(np.float32) / 255
        img_h, img_w, _ = img.shape
        all_imgs.append(img)

        with torch.no_grad():
            ori_size = (img_h, img_w)
            X = torch.FloatTensor(img)[None, ...].permute(0, 3, 1, 2).to(device)
            pred_d = model.infer(X).cpu().numpy()[0, 0, ...]  # metric depth, [H, W]
            assert pred_d.shape[0] == img_h, f"{img.shape}, {pred_d.shape}"
            assert pred_d.shape[1] == img_w, f"{img.shape}, {pred_d.shape}"
            full_pred_depths.append(pred_d)

        out = all_w2c[img_i, :] @ h_pt

        im_pt = all_K[img_i, ...] @ out[:3, :]

        depth = im_pt[2, :].copy()
        im_pt = im_pt / im_pt[2:, :]  # [3, #pt]

        # True value means dynamic
        mask = np.array(PIL.Image.open(mask_f)).astype(np.float32)

        select_idx = np.where(
            (im_pt[0, :] >= 0)
            * (im_pt[0, :] < img_w)
            * (im_pt[1, :] >= 0)
            * (im_pt[1, :] < img_h)
        )[0]

        pts = im_pt[:, select_idx]  # [3, #pt]
        depth = depth[select_idx]

        out = map_coordinates(mask, [pts[1, :], pts[0, :]])
        select_idx = np.where(out < 0.1)[0]  # static areas
        pts = pts[:, select_idx]
        depth = depth[select_idx]
        select_idx = np.where(depth > 1e-3)[0]
        pts = pts[:, select_idx]
        depth = depth[select_idx]

        pred_depth = map_coordinates(pred_d, [pts[1, :], pts[0, :]])

        mvs_depths.append(depth)
        pred_depths.append(pred_depth)
        pts_list.append(pts)

    print("calculating scale and shift")

    all_disp_indiv_scales_med = []
    all_disp_indiv_shifts_med = []

    all_disp_indiv_scales_trim = []
    all_disp_indiv_shifts_trim = []

    all_flag_trim = []

    for idx in tqdm.tqdm(range(n_all_frames)):
        # Align predicted depth wrt MVS depth.

        nn_depth = pred_depths[idx]
        nn_full_depth = full_pred_depths[idx]
        mvs_depth = mvs_depths[idx]

        assert np.min(nn_depth) >= 0, f"{idx}, {np.min(nn_depth)}, {np.max(nn_depth)}"
        assert (
            np.min(nn_full_depth) >= 0
        ), f"{idx}, {np.min(nn_depth)}, {np.max(nn_depth)}"
        assert (
            np.min(mvs_depth) >= 0
        ), f"{idx}, {np.min(mvs_depth)}, {np.max(mvs_depth)}"

        # nn_disp = np.clip(1 / (nn_depth + TINY_VAL), 0.01, None)
        # mvs_disp = np.clip(1 / (mvs_depth + TINY_VAL), 0.01, None)

        # Scale and shift should be estimated in inverse depth domain.
        # For the reason, see Eq. (1) of https://arxiv.org/abs/1904.11112
        nn_disp = 1 / (nn_depth + TINY_VAL)
        nn_full_disp = 1 / (nn_full_depth + TINY_VAL)
        mvs_disp = 1 / (mvs_depth + TINY_VAL)

        # estimate raw value median, then normalize by removing the medians
        nn_disp_median = np.median(nn_disp)
        mvs_disp_median = np.median(mvs_disp)

        # Eq. (5) of https://arxiv.org/abs/1907.01341
        nn_disp_shifted = nn_disp - nn_disp_median
        mvs_disp_shifted = mvs_disp - mvs_disp_median

        # estimate median scales
        disp_indiv_scale_med = np.median(
            mvs_disp_shifted / (nn_disp_shifted + TINY_VAL)
        )
        if disp_indiv_scale_med < 0:
            # We should not change the relative order of predicted depth
            disp_indiv_scale_med = 0.0
        all_disp_indiv_scales_med.append(disp_indiv_scale_med)

        # estimate median shift
        if True:
            disp_indiv_shift_med = np.median(mvs_disp - nn_disp * disp_indiv_scale_med)
        else:
            disp_indiv_shift_med = np.mean(mvs_disp - nn_disp * disp_indiv_scale_med)
        all_disp_indiv_shifts_med.append(disp_indiv_shift_med)

        nn_disp_scale = np.mean(np.abs(nn_disp_shifted))
        mvs_disp_scale = np.mean(np.abs(mvs_disp_shifted))

        nn_disp_normalized = nn_disp_shifted / (nn_disp_scale + TINY_VAL)
        mvs_disp_normalized = mvs_disp_shifted / (mvs_disp_scale + TINY_VAL)

        mvs_nn_disp_normalized_diff = np.abs(nn_disp_normalized - mvs_disp_normalized)
        disp_diff_thres = np.quantile(mvs_nn_disp_normalized_diff, 0.8)

        flag_trim = mvs_nn_disp_normalized_diff <= disp_diff_thres
        all_flag_trim.append(flag_trim)

        nn_disp_trim = nn_disp[flag_trim]
        mvs_disp_trim = mvs_disp[flag_trim]

        nn_disp_shifted_trim = nn_disp_shifted[flag_trim]
        mvs_disp_shifted_trim = mvs_disp_shifted[flag_trim]

        disp_indiv_scale_trim = np.median(
            mvs_disp_shifted_trim / (nn_disp_shifted_trim + TINY_VAL)
        )
        if disp_indiv_scale_trim < 0:
            # We should not change the relative order of predicted depth
            disp_indiv_scale_trim = 0.0
        all_disp_indiv_scales_trim.append(disp_indiv_scale_trim)

        if True:
            disp_indiv_shift_trim = np.median(
                mvs_disp_trim - nn_disp_trim * disp_indiv_scale_trim
            )
        else:
            disp_indiv_shift_trim = np.mean(
                mvs_disp_trim - nn_disp_trim * disp_indiv_scale_trim
            )
        all_disp_indiv_shifts_trim.append(disp_indiv_shift_trim)

        # scale-shift-invariant loss:
        # - https://gist.github.com/dvdhfnr/732c26b61a0e63a0abc8a5d769dbebd0
        # - https://github.com/isl-org/MiDaS/issues/2#issuecomment-511753522

    disp_share_scale_med = np.mean(all_disp_indiv_scales_med)
    disp_share_shift_med = np.mean(all_disp_indiv_shifts_med)

    disp_share_scale_trim = np.mean(all_disp_indiv_scales_trim)
    disp_share_shift_trim = np.mean(all_disp_indiv_shifts_trim)

    print(
        "\nfinal_disp_scale/shift median: ",
        disp_share_scale_med,
        disp_share_shift_med,
        "\n",
    )

    print(
        "final_disp_scale/shift trim: ",
        disp_share_scale_trim,
        disp_share_shift_trim,
        "\n",
    )

    all_mae_med_share = []
    all_mae_med_indiv = []
    all_mae_trim_share = []
    all_mae_trim_indiv = []

    all_me_med_share = []
    all_me_med_indiv = []
    all_me_trim_share = []
    all_me_trim_indiv = []

    for img_i in tqdm.tqdm(range(n_all_frames)):
        flag_trim = all_flag_trim[img_i]
        mvs_depth_trim = mvs_depths[img_i][flag_trim]
        nn_disp_trim = 1 / (pred_depths[img_i][flag_trim] + TINY_VAL)

        diff_med_share = mvs_depth_trim - 1 / (
            nn_disp_trim * disp_share_scale_med + disp_share_shift_med
        )
        diff_med_indiv = mvs_depth_trim - 1 / (
            nn_disp_trim * all_disp_indiv_scales_med[img_i]
            + all_disp_indiv_shifts_med[img_i]
        )

        diff_trim_share = mvs_depth_trim - 1 / (
            nn_disp_trim * disp_share_scale_trim + disp_share_shift_trim
        )
        diff_trim_indiv = mvs_depth_trim - 1 / (
            nn_disp_trim * all_disp_indiv_scales_trim[img_i]
            + all_disp_indiv_shifts_trim[img_i]
        )

        # mean absolute error
        mae_med_share = np.mean(np.abs(diff_med_share))
        mae_med_indiv = np.mean(np.abs(diff_med_indiv))
        mae_trim_share = np.mean(np.abs(diff_trim_share))
        mae_trim_indiv = np.mean(np.abs(diff_trim_indiv))

        all_mae_med_share.append(mae_med_share)
        all_mae_med_indiv.append(mae_med_indiv)
        all_mae_trim_share.append(mae_trim_share)
        all_mae_trim_indiv.append(mae_trim_indiv)

        # mean error: this may be better since we prefer predicted depth centered around mvs depth
        me_med_share = np.mean(diff_med_share)
        me_med_indiv = np.mean(diff_med_indiv)
        me_trim_share = np.mean(diff_trim_share)
        me_trim_indiv = np.mean(diff_trim_indiv)

        all_me_med_share.append(me_med_share)
        all_me_med_indiv.append(me_med_indiv)
        all_me_trim_share.append(me_trim_share)
        all_me_trim_indiv.append(me_trim_indiv)

        save_dict = {
            # scale, shift median
            "disp_indiv_scale_med": all_disp_indiv_scales_med[img_i],
            "disp_indiv_shift_med": all_disp_indiv_shifts_med[img_i],
            "disp_share_scale_med": disp_share_scale_med,
            "disp_share_shift_med": disp_share_shift_med,
            # scale, shift trim
            "disp_indiv_scale_trim": all_disp_indiv_scales_trim[img_i],
            "disp_indiv_shift_trim": all_disp_indiv_shifts_trim[img_i],
            "disp_share_scale_trim": disp_share_scale_trim,
            "disp_share_shift_trim": disp_share_shift_trim,
            # depth for sparse point cloud
            "sparse_pcl": pts3d,
            "proj_pcl": pts_list[img_i],
            "pcl_depth_mvs": mvs_depths[img_i],
            "pcl_depth_pred": pred_depths[img_i],
            # full depth
            "depth_pred": full_pred_depths[img_i],
            "depth_is_disp": False,
            # mean absolute error
            "mae_med_share": mae_med_share,
            "mae_med_indiv": mae_med_indiv,
            "mae_trim_share": mae_trim_share,
            "mae_trim_indiv": mae_trim_indiv,
            # mean error
            "me_med_share": me_med_share,
            "me_med_indiv": me_med_indiv,
            "me_trim_share": me_trim_share,
            "me_trim_indiv": me_trim_indiv,
        }

        with open(depth_dir / f"{img_i:05d}.npz", "wb") as f:
            np.savez(f, **save_dict)

        if not args.save_space:
            tmp_disp = 1 / (full_pred_depths[img_i] + TINY_VAL)

            # for median
            vis_pcl(
                np.copy(tmp_disp),
                all_disp_indiv_scales_med[img_i],
                all_disp_indiv_shifts_med[img_i],
                all_imgs[img_i],
                all_K[img_i],
                all_c2w[img_i],
                vis_dir_indiv_med / f"{img_i:05d}.ply",
            )
            vis_pcl(
                np.copy(tmp_disp),
                disp_share_scale_med,
                disp_share_shift_med,
                all_imgs[img_i],
                all_K[img_i],
                all_c2w[img_i],
                vis_dir_share_med / f"{img_i:05d}.ply",
            )

            # for trim
            vis_pcl(
                np.copy(tmp_disp),
                all_disp_indiv_scales_trim[img_i],
                all_disp_indiv_shifts_trim[img_i],
                all_imgs[img_i],
                all_K[img_i],
                all_c2w[img_i],
                vis_dir_indiv_trim / f"{img_i:05d}.ply",
            )
            vis_pcl(
                np.copy(tmp_disp),
                disp_share_scale_trim,
                disp_share_shift_trim,
                all_imgs[img_i],
                all_K[img_i],
                all_c2w[img_i],
                vis_dir_share_trim / f"{img_i:05d}.ply",
            )

    print("\nmean absolute error:")
    print("all_mae_med_share: ", np.mean(all_mae_med_share))
    print("all_mae_med_indiv: ", np.mean(all_mae_med_indiv))
    print("all_mae_trim_share: ", np.mean(all_mae_trim_share))
    print("all_mae_trim_indiv: ", np.mean(all_mae_trim_indiv), "\n")

    print("\nmean error:")
    print("all_me_med_share: ", np.mean(all_me_med_share))
    print("all_me_med_indiv: ", np.mean(all_me_med_indiv))
    print("all_me_trim_share: ", np.mean(all_me_trim_share))
    print("all_me_trim_indiv: ", np.mean(all_me_trim_indiv), "\n")
