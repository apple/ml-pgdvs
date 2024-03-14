import os
import tqdm
import pathlib
import hydra
import hydra.utils
import numpy as np
from collections import OrderedDict

import torch

from pgdvs.utils.training import disabled_train
from pgdvs.models.gnt.projector import Projector
from pgdvs.models.gnt.ray_sampler import (
    sample_along_camera_ray,
    sample_fine_pts,
)
from pgdvs.utils.vis_utils import draw_cam_mesh
from pgdvs.models.gnt.common import TINY_NUMBER, HUGE_NUMBER


class BaseRenderer(torch.nn.Module):
    def __init__(self, *, model_cfg):
        super().__init__()

        self.projector = Projector()

        # Workaround 1 in https://github.com/facebookresearch/hydra/issues/1950#issuecomment-1009610535
        self.model = hydra.utils.instantiate([model_cfg])[0]

    def forward(
        self,
        *,
        ray_batch,
        chunk_size,
        inv_uniform=False,
        n_coarse_samples_per_ray,
        n_fine_samples_per_ray=0,
        flag_deterministic=False,
        use_dyn_mask=False,
        render_stride=1,
        ret_view_entropy=False,
        ret_view_std=False,
        debug_epipolar=False,
        disable_tqdm=False,
    ):
        # ray_batch:
        # - ray_o: [BxHxW, 3] or [BxHxW, #src, #flow, 3]
        # - ray_d: [BxHxW, 3] or [BxHxW, #src, #flow, 3]
        # - depth_range: [B, 2]
        # - camera: [B, 34]
        # - rgb: None
        # - src_rgbs: [B, #src, H, W, 3]
        # - src_cameras: [B, #src, 34]

        all_ret = OrderedDict(
            [("outputs_coarse", OrderedDict()), ("outputs_fine", OrderedDict())]
        )

        n_rays = ray_batch["ray_o"].shape[0]

        if debug_epipolar:
            debug_idx = self._vis_for_debug_epipolar(ray_batch, render_stride)

        # - in: [#src, 3, H, W]
        # - out: tupel(coarse, fine), [#src, #feat, H/4, W/4]
        tmp_b, tmp_n_src, tmp_h, tmp_w, _ = ray_batch["src_rgbs"].shape
        tmp_in = (
            ray_batch["src_rgbs"]
            .permute(0, 1, 4, 2, 3)
            .reshape((tmp_b * tmp_n_src, 3, tmp_h, tmp_w))
        )
        raw_featmaps = self.model.feature_net(
            tmp_in
        )  # tuple of (coarse, fine), each of [B x #src, feat_dim, H, W]

        featmaps = [
            raw_featmaps[0].reshape(
                [tmp_b, tmp_n_src] + list(raw_featmaps[0].shape[1:])
            ),
            raw_featmaps[1].reshape(
                [tmp_b, tmp_n_src] + list(raw_featmaps[1].shape[1:])
            ),
        ]  # [B, #src, feat_dim, H, W]

        if chunk_size < 0:
            chunk_size = n_rays

        for chunk_i in tqdm.tqdm(
            range(0, n_rays, chunk_size), disable=disable_tqdm, desc="gnt"
        ):
            if debug_epipolar:
                # NOTE: DEBUG
                chunk_i = debug_idx
                chunk_size = 1

            chunk = OrderedDict()
            for k in ray_batch:
                tmp_k_list = [
                    "ray_o",
                    "ray_d",
                    # "view_ray_o",
                    # "view_ray_d",
                    "view_uv",
                    "batch_refs",
                ]
                if ray_batch["depth_range_per_ray"]:
                    tmp_k_list.append("depth_range")
                if k in tmp_k_list:
                    chunk[k] = ray_batch[k][chunk_i : chunk_i + chunk_size]
                elif ray_batch[k] is not None:
                    chunk[k] = ray_batch[k]
                else:
                    chunk[k] = None

            ret = self.render_func(
                ray_batch=chunk,
                featmaps=featmaps,
                n_coarse_samples_per_ray=n_coarse_samples_per_ray,
                n_fine_samples_per_ray=n_fine_samples_per_ray,
                inv_uniform=inv_uniform,
                flag_deterministic=flag_deterministic,
                use_dyn_mask=use_dyn_mask,
                ret_view_entropy=ret_view_entropy,
                ret_view_std=ret_view_std,
                debug_epipolar=debug_epipolar,
            )

            # handle both coarse and fine outputs
            # cache chunk results on cpu
            if chunk_i == 0:
                for k in ret["outputs_coarse"]:
                    if ret["outputs_coarse"][k] is not None:
                        all_ret["outputs_coarse"][k] = []

                if ret["outputs_fine"] is None:
                    all_ret["outputs_fine"] = None
                else:
                    for k in ret["outputs_fine"]:
                        if ret["outputs_fine"][k] is not None:
                            all_ret["outputs_fine"][k] = []

            for k in ret["outputs_coarse"]:
                if ret["outputs_coarse"][k] is not None:
                    all_ret["outputs_coarse"][k].append(ret["outputs_coarse"][k])

            if ret["outputs_fine"] is not None:
                for k in ret["outputs_fine"]:
                    if ret["outputs_fine"][k] is not None:
                        all_ret["outputs_fine"][k].append(ret["outputs_fine"][k])

        rgb_strided = torch.ones((ray_batch["raw_h"], ray_batch["raw_w"], 3))[
            ::render_stride, ::render_stride, :
        ]

        # merge chunk results and reshape
        for k in all_ret["outputs_coarse"]:
            if k == "random_sigma":
                continue
            # print("\nk: ", k, torch.cat(all_ret["outputs_coarse"][k], dim=0).shape, "\n")
            tmp = torch.cat(all_ret["outputs_coarse"][k], dim=0).reshape(
                (tmp_b, rgb_strided.shape[0], rgb_strided.shape[1], -1)
            )  # [#pix, 3] -> [h, w, 3]
            all_ret["outputs_coarse"][k] = tmp  # .squeeze()

        # TODO: if invalid: replace with white
        # all_ret["outputs_coarse"]["rgb"][all_ret["outputs_coarse"]["mask"] == 0] = 1.0
        if all_ret["outputs_fine"] is not None:
            for k in all_ret["outputs_fine"]:
                if k == "random_sigma":
                    continue
                tmp = torch.cat(all_ret["outputs_fine"][k], dim=0).reshape(
                    (tmp_b, rgb_strided.shape[0], rgb_strided.shape[1], -1)
                )

                all_ret["outputs_fine"][k] = tmp  # .squeeze()

        return all_ret

    def render_func(
        self,
        *,
        ray_batch,
        featmaps,
        n_coarse_samples_per_ray,
        n_fine_samples_per_ray,
        inv_uniform=False,
        flag_deterministic=False,
        use_dyn_mask=False,
        ret_view_entropy=False,
        ret_view_std=False,
        debug_epipolar=False,
    ):
        ret = self.render_rays(
            ray_batch=ray_batch,
            featmaps=featmaps,
            n_coarse_samples_per_ray=n_coarse_samples_per_ray,
            n_fine_samples_per_ray=n_fine_samples_per_ray,
            inv_uniform=inv_uniform,
            flag_deterministic=flag_deterministic,
            use_dyn_mask=use_dyn_mask,
            ret_view_entropy=ret_view_entropy,
            ret_view_std=ret_view_std,
            debug_epipolar=debug_epipolar,
        )
        return ret

    def render_rays(
        self,
        *,
        ray_batch,
        featmaps,
        n_coarse_samples_per_ray,
        n_fine_samples_per_ray,
        inv_uniform=False,
        flag_deterministic=False,
        use_dyn_mask=False,
        ret_view_entropy=False,
        ret_view_std=False,
        debug_epipolar=False,
    ):
        ray_o, ray_d = (
            ray_batch["ray_o"],
            ray_batch["ray_d"],
        )  # both are [#pix, 3] or [#pix, #src, #flow, 3]

        n_src_views = ray_batch["src_rgbs"].shape[1]

        # pts: [#ray, #sample, 3] or [#ray, #sample, #src, #flow, 3]
        # z_vals: [#ray, #sample] or [#ray, #sample, #src, #flow]

        if ray_batch["depth_range_per_ray"]:
            tmp_depth_range = ray_batch["depth_range"]
        else:
            tmp_depth_range = ray_batch["depth_range"][ray_batch["batch_refs"], ...]
        pts, z_vals = sample_along_camera_ray(
            ray_o=ray_o,
            ray_d=ray_d,
            depth_range=tmp_depth_range,
            N_samples=n_coarse_samples_per_ray,
            inv_uniform=inv_uniform,
            det=flag_deterministic,
        )

        debug_epipolar_infos = None
        if debug_epipolar:
            debug_epipolar_infos = self._prepare_debug_epipolar_infos(pts, ray_batch)

        rgb_feat, ray_diff, mask_inbound, mask_dy, mask = self.project_true_batch(
            ray_o=ray_o,
            featmaps=featmaps[0],
            pts=pts,
            ray_batch=ray_batch,
            debug_epipolar=debug_epipolar,
            debug_epipolar_infos=debug_epipolar_infos,
            use_dyn_mask=use_dyn_mask,
        )  # [#ray, #sample, #src, x], x= 3 + d (rgb_feat); 4 (ray_diff); 1 (mask)

        rgb, extra_outs = self.model.net_coarse(
            rgb_feat,
            ray_diff,
            mask,
            pts,
            ray_d,
            ret_view_entropy=ret_view_entropy,
            ret_view_std=ret_view_std,
        )  # [#ray, 3] or [#ray, 4]

        rgb, weights = rgb[:, 0:3], rgb[:, 3:]  # weights: [#ray, #sample]
        depth_map = torch.sum(weights * z_vals, dim=-1)  # [#ray, ]

        inbound_cnt_coarse = torch.sum(
            weights * mask_inbound[..., 0].sum(dim=2).float() / n_src_views, dim=1
        )  # [#ray, #sample]. True indicates projections are out-of-bound on all source views

        dyn_cnt_coarse = torch.sum(
            weights * mask_dy[..., 0].sum(dim=2).float() / n_src_views, dim=1
        )  # [#ray, #sample]. True indicates projections to dynamic mask

        ret = {"outputs_coarse": None, "outputs_fine": None}

        ret["outputs_coarse"] = {
            "rgb": rgb,
            "weights": weights,
            "depth": depth_map,
            "inbound_cnt": inbound_cnt_coarse,
            "dyn_cnt": dyn_cnt_coarse,
        }

        if ret_view_entropy:
            view_entropy = extra_outs["view_entropy"]  # [#ray, #sample, #layer]
            view_entropy = torch.sum(
                weights[..., None] * view_entropy, dim=1
            )  # [#ray, #layers]

            ret["outputs_coarse"]["view_entropy"] = view_entropy

        if ret_view_std:
            view_std = extra_outs["view_std"]  # [#ray, #sample, #layer]
            view_std_normalized = extra_outs[
                "view_std_normalized"
            ]  # [#ray, #sample, #layer]

            view_std = torch.sum(
                weights[..., None] * view_std, dim=1
            )  # [#ray, #layers]
            view_std_normalized = torch.sum(
                weights[..., None] * view_std_normalized, dim=1
            )  # [#ray, #layers]

            ret["outputs_coarse"]["view_std"] = view_std
            ret["outputs_coarse"]["view_std_normalized"] = view_std_normalized

        if n_fine_samples_per_ray > 0:
            assert (
                ray_o.ndim == 2
            ), f"Do not support ray shift for importance sampling yet."

            # detach since we would like to decouple the coarse and fine networks
            weights = (
                ret["outputs_coarse"]["weights"].clone().detach()
            )  # [#ray, #sample]
            pts, z_vals = sample_fine_pts(
                inv_uniform,
                n_fine_samples_per_ray,
                flag_deterministic,
                n_coarse_samples_per_ray,
                ray_batch,
                weights,
                z_vals,
            )

            (
                rgb_feat_sampled,
                ray_diff,
                mask_inbound,
                mask_dy,
                mask,
            ) = self.project_true_batch(
                ray_o=ray_o,
                featmaps=featmaps[1],
                pts=pts,
                ray_batch=ray_batch,
                debug_epipolar=debug_epipolar,
                debug_epipolar_infos=debug_epipolar_infos,
                use_dyn_mask=use_dyn_mask,
            )  # [#ray, #sample, #src, x], x= 3 + d (rgb_feat); 4 (ray_diff); 1 (mask)

            if self.model.single_net:
                rgb, extra_outs = self.model.net_coarse(
                    rgb_feat_sampled,
                    ray_diff,
                    mask,
                    pts,
                    ray_d,
                    ret_view_entropy=ret_view_entropy,
                    ret_view_std=ret_view_std,
                )
            else:
                rgb, extra_outs = self.model.net_fine(
                    rgb_feat_sampled,
                    ray_diff,
                    mask,
                    pts,
                    ray_d,
                    ret_view_entropy=ret_view_entropy,
                    ret_view_std=ret_view_std,
                )

            rgb, weights = rgb[:, 0:3], rgb[:, 3:]
            depth_map = torch.sum(weights * z_vals, dim=-1)

            inbound_cnt_fine = torch.sum(
                weights * mask_inbound[..., 0].sum(dim=2).float() / n_src_views, dim=1
            )  # [#ray, #sample]. True indicates projections are out-of-bound on all source views

            dyn_cnt_fine = torch.sum(
                weights * mask_dy[..., 0].sum(dim=2).float() / n_src_views, dim=1
            )  # [#ray, #sample]. True indicates projections to dynamic mask

            ret["outputs_fine"] = {
                "rgb": rgb,
                "weights": weights,
                "depth": depth_map,
                "inbound_cnt": inbound_cnt_fine,
                "dyn_cnt": dyn_cnt_fine,
            }

            if ret_view_entropy:
                view_entropy = extra_outs["view_entropy"]  # [#ray, #sample, #layer]
                view_entropy = torch.sum(
                    weights[..., None] * view_entropy, dim=1
                )  # [#ray, #layers]

                ret["outputs_fine"]["view_entropy"] = view_entropy

            if ret_view_std:
                view_std = extra_outs["view_std"]  # [#ray, #sample, #layer]
                view_std_normalized = extra_outs[
                    "view_std_normalized"
                ]  # [#ray, #sample, #layer]

                view_std = torch.sum(
                    weights[..., None] * view_std, dim=1
                )  # [#ray, #layers]
                view_std_normalized = torch.sum(
                    weights[..., None] * view_std_normalized, dim=1
                )  # [#ray, #layers]

                ret["outputs_fine"]["view_std"] = view_std
                ret["outputs_fine"]["view_std_normalized"] = view_std_normalized

        return ret

    def project_true_batch(
        self,
        *,
        ray_o,
        featmaps,
        pts,
        ray_batch,
        debug_epipolar,
        debug_epipolar_infos,
        use_dyn_mask=False,
    ):
        # NOTE: to deal with true batch
        unique_batch_refs = torch.unique(ray_batch["batch_refs"])
        batch_idxs = torch.arange(ray_batch["batch_refs"].shape[0]).to(pts.device)

        rgb_feat = []
        ray_diff = []
        mask_inbound = []
        mask_invalid = []
        mask = []
        shuffled_batch_idxs = []

        for unique_i in torch.arange(unique_batch_refs.shape[0]):
            unique_batch_idx = unique_batch_refs[unique_i]
            unique_mask = ray_batch["batch_refs"] == unique_batch_idx

            shuffled_batch_idxs.append(batch_idxs[unique_mask])

            masked_pts = pts[unique_mask, ...]

            if use_dyn_mask:
                train_invalid_masks = ray_batch["src_invalid_masks"][
                    unique_batch_idx : (unique_batch_idx + 1), ...
                ]
            else:
                train_invalid_masks = None

            proj_ret = self.projector.compute(
                xyz=masked_pts,
                query_camera=ray_batch["camera"][
                    unique_batch_idx : (unique_batch_idx + 1), ...
                ],
                train_imgs=ray_batch["src_rgbs"][
                    unique_batch_idx : (unique_batch_idx + 1), ...
                ],
                train_cameras=ray_batch["src_cameras"][
                    unique_batch_idx : (unique_batch_idx + 1), ...
                ],
                train_invalid_masks=train_invalid_masks,
                featmaps=featmaps[unique_batch_idx, ...],
                debug_epipolar=debug_epipolar,
                debug_epipolar_infos=debug_epipolar_infos,
            )  # [#ray, #sample, #src, x], x= 3 + d (rgb_feat); 4 (ray_diff); 1 (mask)

            rgb_feat.append(proj_ret["rgb_feat"])
            ray_diff.append(proj_ret["ray_diff"])
            mask_inbound.append(proj_ret["mask_inbound"])
            mask.append(proj_ret["mask"])
            if train_invalid_masks is not None:
                mask_invalid.append(proj_ret["mask_invalid"])
            else:
                mask_invalid.append(torch.zeros_like(proj_ret["mask_inbound"]))

        sorted_batch_idxs = torch.argsort(torch.cat(shuffled_batch_idxs, dim=0))

        rgb_feat = torch.cat(rgb_feat, dim=0)[sorted_batch_idxs, ...]
        ray_diff = torch.cat(ray_diff, dim=0)[sorted_batch_idxs, ...]
        mask_inbound = torch.cat(mask_inbound, dim=0)[sorted_batch_idxs, ...]
        mask_invalid = torch.cat(mask_invalid, dim=0)[sorted_batch_idxs, ...]
        mask = torch.cat(mask, dim=0)[sorted_batch_idxs, ...]

        return rgb_feat, ray_diff, mask_inbound, mask_invalid, mask

    def _vis_for_debug_epipolar(self, ray_batch, render_stride):
        # NOTE: DBEUG
        # assert (
        #     ray_batch["depth_range"].shape[0] == 1
        # ), f"{ray_batch['depth_range'].shape}"

        print("\ndebug_dir: ", ray_batch["debug_dir"], "\n")
        print("\ndebug_row, debug_col: ", ray_batch["debug_pix_coord"], "\n")
        debug_row, debug_col = ray_batch["debug_pix_coord"]
        debug_row = int(debug_row)
        debug_col = int(debug_col)
        tmp = np.ones([ray_batch["raw_h"], ray_batch["raw_w"], 3])[
            ::render_stride, ::render_stride, :
        ]
        tmp_h, tmp_w, _ = tmp.shape

        debug_idx = debug_row * tmp_w + debug_col

        print("\nrender_single_image: ", ray_batch["camera"][:, :2])
        tgt_w2c = np.linalg.inv(
            ray_batch["camera"][0, -16:].reshape(4, 4).cpu().numpy()
        )
        draw_cam_mesh(tgt_w2c, ray_batch["debug_dir"] / "tgt_cam.ply", tmp_coord=0.1)

        print("\n render_pose: ", ray_batch["camera"][0, -16:].reshape(4, 4), "\n")

        for tmp_i in range(ray_batch["src_cameras"].shape[1]):
            tmp_src_w2c = np.linalg.inv(
                ray_batch["src_cameras"][0, tmp_i, -16:].reshape(4, 4).cpu().numpy()
            )
            draw_cam_mesh(
                tmp_src_w2c,
                ray_batch["debug_dir"] / f"src_cam_{tmp_i:03d}.ply",
                tmp_coord=0.1,
            )
        return debug_idx

    def _prepare_debug_epipolar_infos(self, pts, ray_batch):
        # fmt :off
        # NOTE: DEBUG
        print("\n[render ray] [debug epipolar] pts: ", pts.shape)

        from pgdvs.utils.vis_utils import draw_ray_pcl

        debug_tgt_img = (ray_batch["rgb"].cpu().numpy()[0, ...] * 255).astype(np.uint8)

        print(
            "\n[render ray] [debug epipolar] debug_tgt_img: ",
            ray_batch["rgb"].shape,
            "\n",
        )

        debug_epipolar_infos = {
            "tgt_img": debug_tgt_img,
            "tgt_pix_coord": ray_batch["debug_pix_coord"],  # [2, ]
            "vert_rgbs": {},
            "debug_dir": ray_batch["debug_dir"],
        }

        if pts.ndim == 3:
            # Here: [#ray, #sample, 3]
            print(
                'ray_batch["depth_range"]: ',
                ray_batch["depth_range"],
                torch.min(pts.reshape((-1, 3)), dim=0).values,
                torch.max(pts.reshape((-1, 3)), dim=0).values,
                "\n",
            )
            tmp_px_ray_pt_rgbs = draw_ray_pcl(
                0, pts[0, ...].cpu().numpy(), ray_batch["debug_dir"] / "tgt_ray.ply"
            )
            debug_epipolar_infos["vert_rgbs"] = tmp_px_ray_pt_rgbs
        else:
            raise AttributeError(pts.shape)
        # fmt: on
        return debug_epipolar_infos
