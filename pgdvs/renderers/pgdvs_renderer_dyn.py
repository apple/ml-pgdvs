import logging
import hydra
import hydra.utils
import trimesh
import pathlib
import PIL.Image
import numpy as np

import torch
import pytorch3d
import pytorch3d.utils
import pytorch3d.ops as p3d_ops

from pgdvs.utils.training import disabled_train
from pgdvs.utils.pytorch3d_utils import SimpleShader
from pgdvs.renderers.pgdvs_renderer_base import PGDVSBaseRenderer


DEBUG_DIR = pathlib.Path(__file__).absolute().parent.parent.parent / "debug/dyn_render"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)


LOGGER = logging.getLogger(__name__)

FLAG_DEBUG_EPIPOLAR = False


class PGDVSDynamicRenderer(PGDVSBaseRenderer):
    def __init__(
        self,
        *,
        cfg,
        softsplat_metric_abs_alpha=100.0,
        proj_func=None,
        local_rank=0,
        use_tracker=False,
    ):
        super(PGDVSDynamicRenderer, self).__init__()

        self.cfg = cfg

        self.softsplat_metric_abs_alpha = softsplat_metric_abs_alpha
        assert (
            self.softsplat_metric_abs_alpha >= 0
        ), f"{self.softsplat_metric_abs_alpha}"

        self.proj_func = proj_func

        self.tracker = None
        self.use_tracker = use_tracker

        if self.use_tracker:
            self.tracker = hydra.utils.instantiate(
                self.cfg.tracker,
                ori_rgb_range=self.cfg.rgb_range,
                local_rank=local_rank,
            )
            if isinstance(self.tracker, torch.nn.Module):
                self.tracker = self.tracker.eval()
                self.tracker.train = disabled_train
            self.track_chunk_size = self.tracker.query_chunk_size

    def forward(self, data, ray_batch, render_cfg, for_debug=False, disable_tqdm=False):
        n_b, n_src_temporal, orig_h, orig_w, _ = data["rgb_src_temporal"].shape

        flow_1_to_tgt = []
        dyn_mask_src_1 = []
        rgb_src_1 = []
        rgb_src_2 = []
        flow_src1_to_src2 = []
        mesh_render_info = {
            "rgb": [],
            "mask": [],
            "pcl": [],
            "pcl_rgbs": [],
            "pcl_nn_dist_thres": [],
        }

        for i_b in range(n_b):
            # for i_1, i_2, flow_name in [[0, 1, "flow_fwd"], [1, 0, "flow_bwd"]]:
            for i_1, i_2, flow_name in [[0, 1, "flow_fwd"]]:
                tmp_K_1 = data["flat_cam_src_temporal"][i_b, i_1, 2:18].reshape(
                    (1, 4, 4)
                )
                tmp_c2w_1 = data["flat_cam_src_temporal"][i_b, i_1, 18:34].reshape(
                    (1, 4, 4)
                )

                tmp_K_2 = data["flat_cam_src_temporal"][i_b, i_2, 2:18].reshape((4, 4))
                tmp_c2w_2 = data["flat_cam_src_temporal"][i_b, i_2, 18:34].reshape(
                    (4, 4)
                )

                (tmp_rays_o, tmp_rays_d, tmp_view_uvs, _, _) = self.get_batched_rays(
                    device=data["rgb_src_temporal"].device,
                    batch_size=1,
                    H=orig_h,
                    W=orig_w,
                    render_stride=1,
                    intrinsics=tmp_K_1,
                    c2w=tmp_c2w_1,
                )

                if torch.sum(data["dyn_mask_src_temporal"][i_b, i_1, ...]) > 0:
                    # fmt: off
                    tmp_flow_1_to_tgt, valid_dyn_mask_src1, tmp_mesh_render_info = self.compute_dyn_pcl(
                        dyn_mask_1=data["dyn_mask_src_temporal"][i_b, i_1, ...],  # [H, W, 1]
                        rgb_1=data["rgb_src_temporal"][i_b, i_1, ...],  # [H, W, 3]
                        uvs_1=tmp_view_uvs,  # [HxW, 2]
                        ray_o_1=tmp_rays_o,  # [HxW, 3]
                        ray_d_1=tmp_rays_d,  # [HxW, 3]
                        depth_1=data["depth_src_temporal"][i_b, i_1, ...],  # [H, W, 1]
                        flow_12=data[flow_name][i_b, ...],  # [H, W, 2]
                        flow_12_occ_mask=data[f"{flow_name}_occ_mask"][i_b, ...],  # [H, W, 1]
                        rgb_2=data["rgb_src_temporal"][i_b, i_2, ...],  # [H, W, 3]
                        depth_2=data["depth_src_temporal"][i_b, i_2, ...],  # [H, W, 1]
                        K_2=tmp_K_2,  # [4, 4]
                        c2w_2=tmp_c2w_2,  # [4, 4]
                        flat_cam_tgt=data["flat_cam_tgt"][i_b, ...],  # [34]
                        time_1=data["time_src_temporal"][i_b, i_1],
                        time_2=data["time_src_temporal"][i_b, i_2],
                        time_tgt=data["time_tgt"][i_b, 0],
                        render_cfg=render_cfg,
                        for_debug=for_debug,
                    )
                    # fmt: on

                    flow_1_to_tgt.append(tmp_flow_1_to_tgt)
                    rgb_src_1.append(data["rgb_src_temporal"][i_b, i_1, ...])
                    rgb_src_2.append(data["rgb_src_temporal"][i_b, i_2, ...])
                    flow_src1_to_src2.append(data[flow_name][i_b, ...])
                    dyn_mask_src_1.append(valid_dyn_mask_src1)
                else:
                    flow_1_to_tgt.append(torch.zeros_like(data[flow_name][i_b, ...]))
                    rgb_src_1.append(
                        torch.zeros_like(data["rgb_src_temporal"][i_b, i_1, ...])
                    )
                    rgb_src_2.append(
                        torch.zeros_like(data["rgb_src_temporal"][i_b, i_1, ...])
                    )
                    flow_src1_to_src2.append(data[flow_name][i_b, ...])
                    dyn_mask_src_1.append(
                        torch.zeros_like(data["dyn_mask_src_temporal"][i_b, i_1, ...])
                    )

                    tmp_mesh_render_info = {
                        "pcl": None,
                        "pcl_rgbs": None,
                        "pcl_nn_dist_thres": None,
                        "rgb": torch.zeros_like(rgb_src_1[-1]),
                        "mask": torch.zeros_like(dyn_mask_src_1[-1]),
                    }

                for tmp_k in mesh_render_info:
                    mesh_render_info[tmp_k].append(tmp_mesh_render_info[tmp_k])

        if render_cfg.dyn_render_type == "softsplat":
            dyn_mask_src_1 = torch.stack(dyn_mask_src_1, dim=0).permute(0, 3, 1, 2)
            flow_1_to_tgt = torch.stack(flow_1_to_tgt, dim=0).permute(
                0, 3, 1, 2
            )  # [B, H, W, 2] -> [B, 2, H, W]
            rgb_src_2 = torch.stack(rgb_src_2, dim=0).permute(0, 3, 1, 2)
            flow_src1_to_src2 = torch.stack(flow_src1_to_src2, dim=0).permute(
                0, 3, 1, 2
            )  # * dyn_mask_src_1

            # NOTE: important!
            # We need to enforce static part's RGBs to be black (or something different from the original true color).
            # Since we set flow_1_to_tgt to be zero for static parts,
            # there will be many conflicts that static and dynamic pixels in source view will be mapped to the same location in target view.
            # Softsplat uses color consistency to assign weights (See Eq. 13 in https://arxiv.org/abs/2003.05534),
            # i.e., more consistent, more weigths to the corresponding source pixels.
            # In order to let only source view's dynamic RGB to occupy target pixels,
            # we need to
            # 1) mannually let source views's static regions to be INCONSISTENT;
            # 2) set softsplat_metric_abs_alpha to be reasonably large value.
            rgb_src_1 = torch.stack(rgb_src_1, dim=0).permute(0, 3, 1, 2)
            assert self.cfg.rgb_range == "0_1", f"{self.cfg.rgb_range}"
            rgb_src_1_static_black = rgb_src_1 * dyn_mask_src_1
            rgb_src_1 = rgb_src_1_static_black + torch.clamp(
                torch.randn_like(rgb_src_1), 0.0, 1.0
            ) * (1 - dyn_mask_src_1)

            splat_dyn_img_full, softsplat_metric_src1_to_src2 = self.softsplat_img(
                rgb_src1=rgb_src_1,
                flow_src1_to_tgt=flow_1_to_tgt,
                rgb_src2=rgb_src_2,
                flow_src1_to_src2=flow_src1_to_src2,
                softsplat_metric_src1_to_src2=None,
            )  # [B, 3, H, W]

            render_dyn_mask, _ = self.softsplat_img(
                rgb_src1=dyn_mask_src_1,
                flow_src1_to_tgt=flow_1_to_tgt,
                rgb_src2=rgb_src_2,
                flow_src1_to_src2=flow_src1_to_src2,
                softsplat_metric_src1_to_src2=softsplat_metric_src1_to_src2,
            )  # [B, 1, H, W]

            render_dyn_mask = (render_dyn_mask > 1e-3).float()

            render_dyn_rgb = splat_dyn_img_full * render_dyn_mask
        else:
            render_dyn_rgb = torch.stack(mesh_render_info["rgb"], dim=0).permute(
                0, 3, 1, 2
            )
            render_dyn_mask = torch.stack(mesh_render_info["mask"], dim=0).permute(
                0, 3, 1, 2
            )

        if self.use_tracker:
            assert self.tracker is not None
            base_pcl_info = {
                "pcl": mesh_render_info["pcl"],
                "pcl_rgbs": mesh_render_info["pcl_rgbs"],
                "pcl_nn_dist_thres": mesh_render_info["pcl_nn_dist_thres"],
            }
            render_track_rgb, render_track_mask = self.render_with_track(
                data,
                render_cfg=render_cfg,
                base_pcl_info=base_pcl_info,
                for_debug=for_debug,
                disable_tqdm=disable_tqdm,
            )
        else:
            render_track_rgb = torch.zeros_like(render_dyn_rgb)
            render_track_mask = torch.zeros_like(render_dyn_mask)

        mask_for_track = ((~(render_dyn_mask > 0)) & (render_track_mask > 0)).float()
        render_dyn_rgb_final = (
            1 - mask_for_track
        ) * render_dyn_rgb + mask_for_track * render_track_rgb
        render_dyn_mask_final = (
            (render_dyn_mask > 0) | (render_track_mask > 0)
        ).float()

        render_h = ray_batch["render_h"]
        render_w = ray_batch["render_w"]
        if (render_h != orig_h) or (render_w != orig_w):
            render_dyn_rgb, render_dyn_mask = self.resize_rgb_mask(
                render_dyn_rgb, render_dyn_mask, render_h, render_w
            )
            render_track_rgb, render_track_mask = self.resize_rgb_mask(
                render_track_rgb, render_track_mask, render_h, render_w
            )
            render_dyn_rgb_final, render_dyn_mask_final = self.resize_rgb_mask(
                render_dyn_rgb_final, render_dyn_mask_final, render_h, render_w
            )

        info_dict = {
            "temporal_closest_rgb": render_dyn_rgb,
            "temporal_closest_mask": render_dyn_mask,
            "temporal_track_rgb": render_track_rgb,
            "temporal_track_mask": render_track_mask,
        }

        return render_dyn_rgb_final, render_dyn_mask_final, info_dict

    def resize_rgb_mask(self, rgb, mask, render_h, render_w):
        rgb = torch.nn.functional.interpolate(
            rgb,
            size=(render_h, render_w),
            mode="bicubic",
            align_corners=True,
            antialias=True,
        )
        mask = torch.nn.functional.interpolate(
            mask, size=(render_h, render_w), mode="nearest"
        )
        return rgb, mask

    def render_with_track(self):
        raise NotImplementedError

    def compute_dyn_pcl(
        self,
        *,
        dyn_mask_1,
        rgb_1,
        uvs_1,
        ray_o_1,
        ray_d_1,
        depth_1,
        flow_12,
        flow_12_occ_mask,
        rgb_2,
        depth_2,
        c2w_2,
        K_2,
        flat_cam_tgt,
        time_1,
        time_2,
        time_tgt,
        render_cfg,
        for_debug=False,
    ):
        assert torch.sum(dyn_mask_1) > 0

        raw_h, raw_w, _ = dyn_mask_1.shape
        raw_shape = (
            torch.FloatTensor((raw_w, raw_h)).reshape((1, 2)).to(dyn_mask_1.device)
        )

        flat_dyn_mask_1 = dyn_mask_1.reshape(-1).bool()
        if render_cfg.dyn_render_use_flow_consistency:
            # flow_12_occ_mask: True indicates occlusion
            flat_dyn_mask_1 = ~(flow_12_occ_mask > 0).reshape(-1) & flat_dyn_mask_1
            dyn_mask_1 = flat_dyn_mask_1.reshape((raw_h, raw_w, 1)).float()
        dyn_mask_uv_1 = uvs_1[flat_dyn_mask_1, :2]  # [#pt, 2]
        dyn_mask_flow_12 = flow_12.reshape((-1, 2))[flat_dyn_mask_1, :]  # [#pt, 2]
        uv_dyn_flow_12 = dyn_mask_uv_1 + dyn_mask_flow_12  # [#pt, 2]

        dyn_flow_12_valid_mask = torch.all(
            (uv_dyn_flow_12 >= 0) & (uv_dyn_flow_12 <= raw_shape - 1),
            dim=1,
        )  # [#pt, ]

        dyn_pcl_1 = (ray_o_1 + ray_d_1 * depth_1.reshape((-1, 1)))[flat_dyn_mask_1, :][
            dyn_flow_12_valid_mask, :
        ]  # [#pt, 3]

        # fmt: off
        if for_debug:
            # NOTE: DEBUG
            tmp_debug_pcl_1 = trimesh.PointCloud(
                vertices=dyn_pcl_1.cpu().detach().numpy(),
                colors=rgb_1.reshape((-1, 3))[flat_dyn_mask_1, :][dyn_flow_12_valid_mask, :].cpu().detach().numpy(),
                process=False,
            )
            _ = tmp_debug_pcl_1.export(DEBUG_DIR / f"dyn_pcl_1.ply")
        # fmt: on

        if time_1 == time_2:
            dyn_pcl = dyn_pcl_1
            rgb_flow_12 = rgb_1.reshape((-1, 3))[flat_dyn_mask_1, :][
                dyn_flow_12_valid_mask, :
            ]
        else:
            uv_dyn_flow_12 = uv_dyn_flow_12[dyn_flow_12_valid_mask, :]

            uv_dyn_flow_12_grid = 2 * uv_dyn_flow_12 / raw_shape - 1.0  # [#pt, 2]
            depth_flow_12 = torch.nn.functional.grid_sample(
                depth_2[None, ...].permute(0, 3, 1, 2),  # [B, 1, H, W]
                uv_dyn_flow_12_grid[None, None, ...],  # [1, 1, #pt, 2]
                mode="nearest",
            )[
                0, 0, 0, :
            ]  # [1, 1, 1, #pt] -> [#pt, ]

            rgb_flow_12 = torch.nn.functional.grid_sample(
                rgb_2[None, ...].permute(0, 3, 1, 2),  # [B, 1, H, W]
                uv_dyn_flow_12_grid[None, None, ...],  # [1, 1, #pt, 2]
                mode="bilinear",
            )[
                0, :, 0, :
            ].T  # [1, 3, 1, #pt] -> [#pt, 3]

            uv_dyn_flow_12_homo = torch.cat(
                (uv_dyn_flow_12, torch.ones_like(uv_dyn_flow_12[:, :1])), dim=1
            )  # [#pt, 3]

            dyn_flow_fwd_rays_d = torch.matmul(
                c2w_2[:3, :3],
                torch.matmul(torch.inverse(K_2[:3, :3]), uv_dyn_flow_12_homo.T),
            ).T  # [#pt, 3]
            dyn_flow_fwd_rays_o = c2w_2[:3, 3][None, :].expand(
                dyn_flow_fwd_rays_d.shape[0], -1
            )  # [#pt, 3]

            dyn_pcl_2 = (
                dyn_flow_fwd_rays_o + dyn_flow_fwd_rays_d * depth_flow_12[:, None]
            )  # [#pt, 3]

            # fmt: off
            if for_debug:
                # NOTE: DEBUG
                tmp_debug_pcl_2 = trimesh.PointCloud(
                    vertices=dyn_pcl_2.cpu().detach().numpy(),
                    colors=rgb_flow_12.cpu().detach().numpy(),
                    process=False,
                )
                _ = tmp_debug_pcl_2.export(DEBUG_DIR / f"dyn_pcl_2.ply")
            # fmt: on

            w1 = (time_2 - time_tgt) / (time_2 - time_1)
            w2 = (time_tgt - time_1) / (time_2 - time_1)

            dyn_pcl = w1 * dyn_pcl_1 + w2 * dyn_pcl_2  # [#pt, 3]

        # fmt: off
        if for_debug:
            # NOTE: DEBUG
            tmp_debug_pcl = trimesh.PointCloud(
                vertices=dyn_pcl.cpu().detach().numpy(),
                colors=rgb_flow_12.cpu().detach().numpy(),
                process=False,
            )
            _ = tmp_debug_pcl.export(DEBUG_DIR / f"dyn_pcl.ply")
        # fmt: on

        # removal of outliers
        # - https://github.com/facebookresearch/pytorch3d/issues/511#issuecomment-1152970392
        # - http://www.open3d.org/docs/release/tutorial/geometry/pointcloud_outlier_removal.html
        # - https://pcl.readthedocs.io/en/latest/statistical_outlier.html
        nn_dists, nn_idxs, nn_pts = p3d_ops.knn_points(
            dyn_pcl[None, ...],
            dyn_pcl[None, ...],
            K=(render_cfg.dyn_pcl_outlier_knn + 1),
            return_nn=True,
        )  # nn_dists/idxs: [1, #pts, K]; nn_pts: [1, #pts, K, 3]

        # The 1st distance is always 0 as the nearest is the point itself.
        nn_dists = nn_dists[0, :, 1:]
        nn_idxs = nn_idxs[0, :, 1:]
        nn_pts = nn_pts[0, :, 1:, :]

        # We mimic Open3D's statistical removal
        # https://github.com/isl-org/Open3D/blob/6ddbcd5c9b8bf0b496e4151c7d7766af09e3dba7/cpp/open3d/geometry/PointCloud.cpp#L636-L653
        avg_nn_dist = torch.mean(nn_dists, dim=1)  # [#pt, ]
        dyn_pcl_nn_dist_med = torch.median(avg_nn_dist)
        dyn_pcl_nn_dist_std = torch.std(avg_nn_dist)
        nn_dist_thres = (
            dyn_pcl_nn_dist_med
            + dyn_pcl_nn_dist_std * render_cfg.dyn_pcl_outlier_std_thres
        )

        flag_not_outlier = avg_nn_dist < nn_dist_thres
        assert (
            flag_not_outlier.shape[0] == dyn_pcl.shape[0]
        ), f"{flag_not_outlier.shape}, {dyn_pcl.shape}"
        assert (
            flag_not_outlier.shape[0] == rgb_flow_12.shape[0]
        ), f"{flag_not_outlier.shape}, {rgb_flow_12.shape}"
        assert flag_not_outlier.shape[0] == torch.sum(
            dyn_flow_12_valid_mask
        ), f"{flag_not_outlier.shape}, {torch.sum(dyn_flow_12_valid_mask)}"

        if render_cfg.dyn_pcl_remove_outlier:
            dyn_pcl_clean = dyn_pcl[flag_not_outlier, :]
            rgb_flow_12 = rgb_flow_12[flag_not_outlier, :]

            # fmt: off
            if for_debug:
                # NOTE: DEBUG
                for tmp_i, tmp_pcl in enumerate([dyn_pcl_1, dyn_pcl_2]):
                    tmp_debug_pcl = trimesh.PointCloud(
                        vertices=tmp_pcl[flag_not_outlier, :].cpu().detach().numpy(),
                        colors=rgb_flow_12.cpu().detach().numpy(),
                        process=False,
                    )
                    _ = tmp_debug_pcl.export(DEBUG_DIR / f"dyn_pcl_{tmp_i + 1}_cleaned.ply")
            # fmt: on
        else:
            dyn_pcl_clean = dyn_pcl
            flag_not_outlier = torch.ones(
                (dyn_pcl.shape[0]), dtype=bool, device=dyn_pcl.device
            )

        # fmt: off
        if for_debug:
            # NOTE: DEBUG
            tmp_debug_pcl = trimesh.PointCloud(
                vertices=dyn_pcl_clean.cpu().detach().numpy(),
                colors=rgb_flow_12.cpu().detach().numpy(),
                process=False,
            )
            _ = tmp_debug_pcl.export(DEBUG_DIR / f"dyn_pcl_cleaned.ply")
        # fmt: on

        proj_uvs, _ = self.proj_func(
            dyn_pcl_clean[:, None, :],
            flat_cam_tgt[None, :],
        )  # [#src, #pt, #sample, 2] -> [#pt, 2], #src=#sample=1

        proj_uvs = proj_uvs[0, :, 0, :]  # (row, col) -> (x, y)

        dyn_mask_rows, dyn_mask_cols, _ = torch.nonzero(dyn_mask_1, as_tuple=True)
        valid_dyn_mask_rows = dyn_mask_rows[dyn_flow_12_valid_mask][flag_not_outlier]
        valid_dyn_mask_cols = dyn_mask_cols[dyn_flow_12_valid_mask][flag_not_outlier]

        valid_dyn_mask_1 = torch.zeros_like(dyn_mask_1)
        valid_dyn_mask_1[valid_dyn_mask_rows, valid_dyn_mask_cols, :] = 1.0

        if for_debug:
            # valid_dyn_mask_1: [H, W, 1], float32
            PIL.Image.fromarray(
                (valid_dyn_mask_1[..., 0].cpu().numpy() * 255).astype(np.uint8)
            ).save(DEBUG_DIR / f"valid_dyn_mask_1.png")
            valid_dyn_mask_vert_idxs = -1 * torch.ones_like(dyn_mask_1)
            valid_dyn_mask_vert_idxs[valid_dyn_mask_rows, valid_dyn_mask_cols, :] = (
                torch.arange(valid_dyn_mask_rows.shape[0], device=dyn_mask_1.device)
                .reshape((-1, 1))
                .float()
            )
            np.savez(
                DEBUG_DIR / "valid_dyn_mask_vert_idxs.npz",
                valid_dyn_mask_vert_idxs=valid_dyn_mask_vert_idxs[..., 0].cpu().numpy(),
            )

        flow_1_to_tgt = torch.zeros_like(flow_12)  # [H, W, 2]
        flow_1_to_tgt[valid_dyn_mask_rows, valid_dyn_mask_cols, :] = (
            proj_uvs - dyn_mask_uv_1[dyn_flow_12_valid_mask, :][flag_not_outlier, :]
        )

        mesh_render_info = {
            "pcl": dyn_pcl_clean,
            "pcl_rgbs": rgb_flow_12,
            "pcl_nn_dist_thres": nn_dist_thres,
        }

        if render_cfg.dyn_render_type not in ["softsplat"]:
            if render_cfg.dyn_render_type == "mesh":
                dyn_rgb_tgt, dyn_mask_tgt = self.render_dyn_mesh(
                    rows=valid_dyn_mask_rows,
                    cols=valid_dyn_mask_cols,
                    dyn_mask=valid_dyn_mask_1,
                    dyn_pcl=dyn_pcl_clean,
                    rgbs=rgb_flow_12,
                    flat_cam=flat_cam_tgt,
                    for_debug=for_debug,
                )
            elif render_cfg.dyn_render_type == "pcl":
                dyn_rgb_tgt, dyn_mask_tgt = self.render_dyn_pcl(
                    dyn_mask=valid_dyn_mask_1,
                    dyn_pcl=dyn_pcl_clean,
                    rgbs=rgb_flow_12,
                    flat_cam=flat_cam_tgt,
                    render_cfg=render_cfg,
                    for_debug=for_debug,
                )
            else:
                raise ValueError(render_cfg.dyn_render_type)

            mesh_render_info["rgb"] = dyn_rgb_tgt
            mesh_render_info["mask"] = dyn_mask_tgt
        else:
            mesh_render_info["rgb"] = torch.zeros_like(rgb_1)
            mesh_render_info["mask"] = torch.zeros_like(dyn_mask_1)

        return flow_1_to_tgt, valid_dyn_mask_1, mesh_render_info

    def render_dyn_mesh(
        self, *, rows, cols, dyn_mask, dyn_pcl, rgbs, flat_cam, for_debug=False
    ):
        # We keep the topology of the triangular mesh.
        # Namely, we extract faces based on dynamic mask in temporal source views
        # and use the vertices for the target time.

        h, w, _ = dyn_mask.shape
        vert_idxs_img = -1 * torch.ones(
            (h, w), dtype=torch.long, device=dyn_mask.device
        )
        v_idxs = torch.arange(rows.shape[0], device=dyn_mask.device)
        vert_idxs_img[rows, cols] = v_idxs

        rows_dilated = rows
        cols_dilated = cols

        # Each pixel (row, col) generates two triangles:
        # - (row, col), (row + 1, col), (row + 1, col + 1)
        # - (row, col), (row + 1, col + 1), (row, col + 1)
        face_candidates_1 = torch.stack(
            (
                torch.stack((rows_dilated, cols_dilated), dim=1),
                torch.stack((rows_dilated + 1, cols_dilated), dim=1),
                torch.stack((rows_dilated + 1, cols_dilated + 1), dim=1),
            ),
            dim=1,
        )  # [#face, 3, 2]
        face_candidates_2 = torch.stack(
            (
                torch.stack((rows_dilated, cols_dilated), dim=1),
                torch.stack((rows_dilated + 1, cols_dilated + 1), dim=1),
                torch.stack((rows_dilated, cols_dilated + 1), dim=1),
            ),
            dim=1,
        )  # [#face, 3, 2]

        face_candidates = torch.cat(
            (face_candidates_1, face_candidates_2), dim=0
        )  # [#face, 3, 2]

        flag_in_bound = torch.all(
            (face_candidates[..., 0] >= 0)
            & (face_candidates[..., 0] < h)
            & (face_candidates[..., 1] >= 0)
            & (face_candidates[..., 1] < w),
            dim=1,
        )  # [#faces, ]

        face_candidates = face_candidates[flag_in_bound, ...]  # [#face, 3, 2]

        face_v_idxs = vert_idxs_img[
            face_candidates[..., 0], face_candidates[..., 1]
        ]  # [#faces, 3]

        flag_valid_v = torch.all(face_v_idxs > 0, dim=1)  # [#faces, ]

        if torch.sum(flag_valid_v) == 0:
            mesh_img = torch.zeros_like(dyn_mask).expand(-1, -1, 3)  # [H, W, 3]
            mesh_mask = torch.zeros_like(dyn_mask)  # [H, W, 1]

        else:
            face_v_idxs = face_v_idxs[flag_valid_v, :]  # [#faces, 3]

            if for_debug:
                tmp_debug_mesh = trimesh.Trimesh(
                    vertices=dyn_pcl.detach().cpu().numpy(),
                    faces=face_v_idxs.detach().cpu().numpy(),
                    process=False,
                )
                tmp_debug_mesh.visual.vertex_colors = rgbs.cpu().detach().numpy()
                _ = tmp_debug_mesh.export(
                    DEBUG_DIR / f"torch_mesh.ply", encoding="ascii"
                )

            # render mesh
            K = flat_cam[2:18].reshape((4, 4))
            c2w = flat_cam[18:34].reshape((4, 4))
            w2c = torch.inverse(c2w)

            # lights = pytorch3d.renderer.PointLights(device=w2c.device, location=c2w[None, :3, 3])

            img_size = torch.LongTensor([h, w]).reshape((1, 2))
            cameras_pytorch3d = pytorch3d.utils.cameras_from_opencv_projection(
                w2c[None, :3, :3], w2c[None, :3, 3], K[None, :3, :3], img_size
            )

            raster_settings = pytorch3d.renderer.RasterizationSettings(
                image_size=(h, w),
                blur_radius=0.0,
                faces_per_pixel=1,
                bin_size=0,
            )
            mesh_renderer = pytorch3d.renderer.MeshRenderer(
                rasterizer=pytorch3d.renderer.MeshRasterizer(
                    cameras=cameras_pytorch3d, raster_settings=raster_settings
                ),
                shader=SimpleShader(device=w2c.device),
                # shader=pytorch3d.renderer.HardPhongShader(
                #     device=w2c.device, cameras=cameras_pytorch3d, lights=lights
                # ),
            )

            # https://github.com/facebookresearch/pytorch3d/issues/51#issuecomment-585233728
            dy_mesh = pytorch3d.structures.Meshes(
                dyn_pcl[None, ...],  # [1, #pts, 3]
                faces=face_v_idxs[None, ...],  # [1, #face, 3]
                textures=pytorch3d.renderer.TexturesVertex(
                    verts_features=rgbs[None, ...]
                ),  # [1, #pt, 3]
            )
            mesh_img = mesh_renderer(dy_mesh)[0, :, :, :3]  # [H, W, 3]

            dy_mesh.textures = pytorch3d.renderer.TexturesVertex(
                torch.ones_like(rgbs)[None, ...]
            )
            mesh_mask = (mesh_renderer(dy_mesh)[0, :, :, :1] > 0.0).float()  # [H, W, 1]

            if for_debug:
                PIL.Image.fromarray(
                    (mesh_img.detach().cpu().numpy() * 255).astype(np.uint8)
                ).save(DEBUG_DIR / "dy_mesh_rgb.png")

                PIL.Image.fromarray(
                    (mesh_mask.detach().cpu().numpy()[..., 0] * 255).astype(np.uint8)
                ).save(DEBUG_DIR / "dy_mesh_mask.png")

        return mesh_img, mesh_mask

    def render_dyn_pcl(
        self, *, dyn_mask, dyn_pcl, rgbs, flat_cam, render_cfg, for_debug=False
    ):
        h, w, _ = dyn_mask.shape

        K = flat_cam[2:18].reshape((4, 4))
        c2w = flat_cam[18:34].reshape((4, 4))
        w2c = torch.inverse(c2w)

        if dyn_pcl.shape[0] == 0:
            mesh_img = torch.zeros_like(dyn_mask).expand(-1, -1, 3)  # [H, W, 3]
            mesh_mask = torch.zeros_like(dyn_mask)  # [H, W, 1]
        else:
            img_size = torch.LongTensor([h, w]).reshape((1, 2))
            cameras_pytorch3d = pytorch3d.utils.cameras_from_opencv_projection(
                w2c[None, :3, :3], w2c[None, :3, 3], K[None, :3, :3], img_size
            )

            # for bin size, see https://github.com/facebookresearch/pytorch3d/issues/1064
            raster_settings = pytorch3d.renderer.PointsRasterizationSettings(
                image_size=(h, w),
                radius=render_cfg.dyn_render_pcl_pt_radius,
                points_per_pixel=render_cfg.dyn_render_pcl_pts_per_pixel,
                bin_size=0,
            )

            # Create a points renderer by compositing points using an alpha compositor (nearer points
            # are weighted more heavily). See [1] for an explanation.
            rasterizer = pytorch3d.renderer.PointsRasterizer(
                cameras=cameras_pytorch3d, raster_settings=raster_settings
            )
            point_renderer = pytorch3d.renderer.PointsRenderer(
                rasterizer=rasterizer,
                # compositor=pytorch3d.renderer.AlphaCompositor(
                #     background_color=(0, 0, 0)
                # ),
                compositor=pytorch3d.renderer.NormWeightedCompositor(
                    background_color=(0, 0, 0)
                ),
            )

            dy_mesh = pytorch3d.structures.Pointclouds(
                points=dyn_pcl[None, ...],  # [1, #pts, 3]
                features=rgbs[None, ...],  # [1, #pt, 3]
            )

            mesh_img = point_renderer(dy_mesh)[0, :, :, :3]  # [H, W, 3]

            dy_mesh.features = torch.ones_like(rgbs)[None, ...]
            mesh_mask = (
                point_renderer(dy_mesh)[0, :, :, :1] > 0.0
            ).float()  # [H, W, 1]

        return mesh_img, mesh_mask
