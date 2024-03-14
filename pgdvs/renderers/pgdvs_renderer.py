import logging
import hydra
import hydra.utils
import trimesh
import pathlib
from omegaconf import DictConfig

import torch

from pgdvs.utils.training import disabled_train
from pgdvs.models.gnt.renderer import BaseRenderer as GNTRenderer
from pgdvs.renderers.st_geo_renderer import StaticGeoPointRenderer
from pgdvs.renderers.pgdvs_renderer_base import PGDVSBaseRenderer
from pgdvs.renderers.pgdvs_renderer_dyn import PGDVSDynamicRenderer


DEBUG_DIR = pathlib.Path(__file__).absolute().parent.parent.parent / "debug"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)


LOGGER = logging.getLogger(__name__)

FLAG_DEBUG_EPIPOLAR = False


class PGDVSRenderer(PGDVSBaseRenderer):
    def __init__(
        self,
        cfg: DictConfig,
        *,
        render_cfg,
        flag_debug=False,
        train_static_renderer=False,
        softsplat_metric_abs_alpha=100.0,
        local_rank=0,
    ):
        super(PGDVSRenderer, self).__init__()

        self.cfg = cfg

        self.flag_debug = flag_debug

        self.static_renderer = None
        self.refine_renderer = None

        if self.cfg.static_renderer._target_ is not None:
            self.static_renderer = hydra.utils.instantiate(self.cfg.static_renderer)
            if not train_static_renderer:
                self.static_renderer = self.static_renderer.eval()
                self.static_renderer.train = (
                    disabled_train  # ensure it will never be trained again
                )

        self.softsplat_metric_abs_alpha = softsplat_metric_abs_alpha
        assert (
            self.softsplat_metric_abs_alpha >= 0
        ), f"{self.softsplat_metric_abs_alpha}"

        assert render_cfg.dyn_render_track_temporal in [
            "none",
            "no_tgt",
        ], f"{render_cfg.dyn_render_track_temporal}"

        if render_cfg.dyn_render_track_temporal == "none":
            dyn_renderer_cls = PGDVSDynamicRenderer
        elif render_cfg.dyn_render_track_temporal == "no_tgt":
            from pgdvs.renderers.pgdvs_renderer_dyn_track import (
                PGDVSDynamicTrackRenderer,
            )

            dyn_renderer_cls = PGDVSDynamicTrackRenderer
        else:
            raise ValueError(render_cfg.dyn_render_track_temporal)

        self.dyn_renderer = dyn_renderer_cls(
            cfg=cfg,
            softsplat_metric_abs_alpha=softsplat_metric_abs_alpha,
            proj_func=self.static_renderer.projector.compute_projections,
            local_rank=local_rank,
            use_tracker=render_cfg.dyn_render_track_temporal == "no_tgt",
        )

    def forward(self, data, render_cfg={}, disable_tqdm=False, for_debug=False):
        # - seq_ids [B, 13]
        # - rgb_tgt [B, H, W, 3]
        # - rgb_src_spatial [B, #spatial, H, W, 3]
        # - dyn_rgb_src_spatial [B, #spatial, H, W, 3]
        # - static_rgb_src_spatial [B, #spatial, H, W, 3]
        # - rgb_src_temporal [B, #temporal, H, W, 3]
        # - dyn_rgb_src_temporal [B, #temporal, H, W, 3]
        # - static_rgb_src_temporal [B, #temporal, H, W, 3]
        # - dyn_mask_src_spatial [B, #spatial, H, W, 1]
        # - dyn_mask_src_temporal [B, #temporal, H, W, 1]
        # - flow_fwd [B, H, W, 2]
        # - flow_fwd_occ_mask [B, H, W, 1]
        # - flow_bwd [B, H, W, 2]
        # - flow_bwd_occ_mask [B, H, W, 1]
        # - flat_cam_tgt [B, 34]
        # - flat_cam_src_spatial [B, #spatial, 34]
        # - flat_cam_src_temporal [B, #temporal, 34]
        # - depth_src_temporal [B, #temporal, H, W, 1]
        # - time_tgt [B, 1]
        # - time_src_temporal [B, 2]

        n_b, n_src_temporal, orig_h, orig_w, _ = data["rgb_src_temporal"].shape

        # render static area with GNT
        ray_batch = self.prepare_ray_batch(
            data=data,
            B=n_b,
            H=orig_h,
            W=orig_w,
            render_stride=render_cfg.render_stride,
            render_cfg=render_cfg,
        )

        ret_dict = {}

        if isinstance(self.static_renderer, GNTRenderer):
            if "rgb_gnt" in data:
                static_rgb = data["rgb_gnt"].permute(0, 3, 1, 2)
                ret_dict["static_coarse_rgb"] = static_rgb
            else:
                static_rgb, st_ret_dict = self.forward_st_gnt(
                    data=data,
                    ray_batch=ray_batch,
                    render_cfg=render_cfg,
                    disable_tqdm=disable_tqdm,
                )
                ret_dict.update(st_ret_dict)

            if render_cfg.pure_gnt or render_cfg.pure_gnt_with_dyn_mask:
                ret_dict["combined_rgb"] = static_rgb
                return ret_dict
        elif isinstance(self.static_renderer, StaticGeoPointRenderer):
            static_rgb, st_ret_dict = self.forward_st_geo(
                data=data,
                ray_batch=ray_batch,
                render_cfg=render_cfg,
            )
            ret_dict.update(st_ret_dict)
        else:
            raise TypeError(type(self.static_renderer))

        # process dynamic part
        (render_dyn_rgb, render_dyn_mask, render_dyn_info) = self.dyn_renderer(
            data,
            ray_batch,
            render_cfg,
            for_debug=for_debug,
            disable_tqdm=disable_tqdm,
        )

        ret_dict["render_dyn_rgb"] = render_dyn_rgb
        ret_dict["render_dyn_mask"] = render_dyn_mask
        ret_dict["render_dyn_temporal_closest_rgb"] = render_dyn_info[
            "temporal_closest_rgb"
        ]
        ret_dict["render_dyn_temporal_closest_mask"] = render_dyn_info[
            "temporal_closest_mask"
        ]
        ret_dict["render_dyn_temporal_track_rgb"] = render_dyn_info[
            "temporal_track_rgb"
        ]
        ret_dict["render_dyn_temporal_track_mask"] = render_dyn_info[
            "temporal_track_mask"
        ]

        # combine static and dynamic
        combined_rgb_static = (1 - render_dyn_mask) * static_rgb
        combined_rgb_dyn = render_dyn_mask * render_dyn_rgb
        combined_rgb = combined_rgb_static + combined_rgb_dyn

        ret_dict["render_dyn_rgb"] = render_dyn_rgb
        ret_dict["render_dyn_mask"] = render_dyn_mask
        ret_dict["combined_rgb"] = combined_rgb
        ret_dict["combined_rgb_static"] = combined_rgb_static
        ret_dict["combined_rgb_dyn"] = combined_rgb_dyn

        return ret_dict

    def forward_st_geo(self, *, data, ray_batch, render_cfg):
        static_rgb = []
        static_mask = []
        for i_b in range(data["flat_cam_tgt"].shape[0]):
            tmp_rgb, tmp_mask = self.static_renderer(
                tgt_h=ray_batch["render_h"],
                tgt_w=ray_batch["render_w"],
                flat_tgt_cam=data["flat_cam_tgt"][i_b, ...],
                st_pcl_rgb=data["st_pcl_rgb"][i_b, ...],
                render_cfg=render_cfg,
            )
            static_rgb.append(tmp_rgb)
            static_mask.append(tmp_mask)

        ret_dict = {
            "geo_static_rgb": torch.stack(static_rgb, dim=0).permute(0, 3, 1, 2),
            "geo_static_mask": torch.stack(static_mask, dim=0).permute(0, 3, 1, 2),
        }

        return ret_dict["geo_static_rgb"], ret_dict

    def forward_st_gnt(self, *, data, ray_batch, render_cfg, disable_tqdm=True):
        if render_cfg.pure_gnt:
            assert not render_cfg.gnt_use_dyn_mask
            assert not render_cfg.gnt_use_masked_spatial_src

        if render_cfg.pure_gnt_with_dyn_mask:
            assert render_cfg.gnt_use_dyn_mask
            assert not render_cfg.gnt_use_masked_spatial_src

        n_b, n_src_spatial, orig_h, orig_w, _ = data["rgb_src_temporal"].shape

        ret_dict = {}

        # - rgb: [B, H, W, 3]
        # - weights: [B, H, W, #sample]
        # - depth: [B, H, W, 1]
        # - view_entropy: [B, H, W, 8]
        # - view_std: [B, H, W, 9]
        # - view_std_normalized: [B, H, W, 9]
        if "static_coarse_rgb" not in data:
            static_ret = self.static_renderer(
                ray_batch=ray_batch,
                chunk_size=render_cfg.chunk_size,
                inv_uniform=render_cfg.sample_inv_uniform,
                n_coarse_samples_per_ray=render_cfg.n_coarse_samples_per_ray,
                n_fine_samples_per_ray=render_cfg.n_fine_samples_per_ray,
                use_dyn_mask=render_cfg.gnt_use_dyn_mask,
                flag_deterministic=True,
                render_stride=render_cfg.render_stride,
                ret_view_entropy=True,
                ret_view_std=True,
                debug_epipolar=FLAG_DEBUG_EPIPOLAR,
                disable_tqdm=disable_tqdm,
            )
        else:
            static_ret = {
                "outputs_coarse": {
                    "rgb": data["init_coarse_rgb"],
                    "depth": data["init_coarse_depth"],
                    "inbound_cnt": data["init_coarse_inbound_cnt"],
                    "dyn_cnt": data["init_coarse_dyn_cnt"],
                    "view_entropy": data["init_coarse_view_entropy"],
                    "view_std": data["init_coarse_view_std"],
                    "view_std_normalized": data["init_coarse_view_std_normalized"],
                },
                "outputs_fine": None,
            }

        ret_dict["static_coarse_rgb"] = static_ret["outputs_coarse"]["rgb"].permute(
            0, 3, 1, 2
        )  # [B, 3, H, W]
        ret_dict["static_coarse_depth"] = static_ret["outputs_coarse"]["depth"].permute(
            0, 3, 1, 2
        )
        ret_dict["static_coarse_view_entropy"] = static_ret["outputs_coarse"][
            "view_entropy"
        ].permute(
            0, 3, 1, 2
        )  # [B, #layer, H, W]
        ret_dict["static_coarse_view_std"] = static_ret["outputs_coarse"][
            "view_std"
        ].permute(
            0, 3, 1, 2
        )  # [B, #layer + 1, H, W]
        ret_dict["static_coarse_view_std_normalized"] = static_ret["outputs_coarse"][
            "view_std_normalized"
        ].permute(
            0, 3, 1, 2
        )  # [B, #layer + 1, H, W]
        ret_dict["static_coarse_inbound_cnt"] = static_ret["outputs_coarse"][
            "inbound_cnt"
        ].permute(0, 3, 1, 2)
        ret_dict["static_coarse_oob_mask"] = (
            ret_dict["static_coarse_inbound_cnt"]
            < (render_cfg.mask_oob_n_proj_thres / n_src_spatial)
        ).float()
        if render_cfg.gnt_use_dyn_mask:
            ret_dict["static_coarse_dyn_cnt"] = static_ret["outputs_coarse"][
                "dyn_cnt"
            ].permute(0, 3, 1, 2)
            ret_dict["static_coarse_dyn_mask_any"] = (
                ret_dict["static_coarse_dyn_cnt"] > 0.0
            ).float()
            ret_dict["static_coarse_dyn_mask_all"] = (
                ret_dict["static_coarse_dyn_cnt"] == 1.0
            ).float()
            ret_dict["static_coarse_dyn_mask_thres"] = (
                ret_dict["static_coarse_dyn_cnt"]
                >= (render_cfg.mask_invalid_n_proj_thres / n_src_spatial)
            ).float()

        depth_gnt = ret_dict["static_coarse_depth"]
        mask_oob = ret_dict["static_coarse_oob_mask"]
        static_rgb = ret_dict["static_coarse_rgb"]
        static_view_entropy = ret_dict["static_coarse_view_entropy"]
        static_view_std = ret_dict["static_coarse_view_std"]
        static_view_std_normalized = ret_dict["static_coarse_view_std_normalized"]

        if static_ret["outputs_fine"] is not None:
            ret_dict["static_fine_rgb"] = static_ret["outputs_fine"]["rgb"].permute(
                0, 3, 1, 2
            )  # [B, 3, H, W]
            ret_dict["static_fine_depth"] = static_ret["outputs_fine"]["depth"].permute(
                0, 3, 1, 2
            )
            ret_dict["static_fine_view_entropy"] = static_ret["outputs_fine"][
                "view_entropy"
            ].permute(
                0, 3, 1, 2
            )  # [B, #layer, H, W]
            ret_dict["static_fine_view_std"] = static_ret["outputs_fine"][
                "view_std"
            ].permute(
                0, 3, 1, 2
            )  # [B, #layer + 1, H, W]
            ret_dict["static_fine_view_std_normalized"] = static_ret["outputs_fine"][
                "view_std_normalized"
            ].permute(
                0, 3, 1, 2
            )  # [B, #layer + 1, H, W]
            ret_dict["static_fine_inbound_cnt"] = static_ret["outputs_fine"][
                "inbound_cnt"
            ].permute(0, 3, 1, 2)
            ret_dict["static_fine_oob_mask"] = (
                ret_dict["static_fine_inbound_cnt"]
                < (render_cfg.mask_oob_n_proj_thres / n_src_spatial)
            ).float()
            if render_cfg.gnt_use_dyn_mask:
                ret_dict["static_fine_dyn_cnt"] = static_ret["outputs_fine"][
                    "dyn_cnt"
                ].permute(0, 3, 1, 2)
                ret_dict["static_fine_dyn_mask_any"] = (
                    ret_dict["static_fine_dyn_cnt"] > 0.0
                ).float()
                ret_dict["static_fine_dyn_mask_all"] = (
                    ret_dict["static_fine_dyn_cnt"] == 1.0
                ).float()
                ret_dict["static_fine_dyn_mask_thres"] = (
                    ret_dict["static_fine_dyn_cnt"]
                    >= (render_cfg.mask_invalid_n_proj_thres / n_src_spatial)
                ).float()

            depth_gnt = ret_dict["static_fine_depth"]
            mask_oob = ret_dict["static_fine_oob_mask"]
            static_rgb = ret_dict["static_fine_rgb"]
            static_view_entropy = ret_dict["static_fine_view_entropy"]
            static_view_std = ret_dict["static_fine_view_std"]
            static_view_std_normalized = ret_dict["static_fine_view_std_normalized"]

        return static_rgb, ret_dict

    def prepare_ray_batch(self, *, data, B, H, W, render_stride, render_cfg):
        # flat_cam: 2 + 16 (K) + 16 (c2w)
        tgt_K = data["flat_cam_tgt"][:, 2:18].reshape((B, 4, 4))
        tgt_c2w = data["flat_cam_tgt"][:, 18:34].reshape((B, 4, 4))

        (
            view_rays_o,
            view_rays_d,
            view_uvs,
            batch_refs,
            render_shape,
        ) = self.get_batched_rays(
            device=data["rgb_src_temporal"].device,
            batch_size=B,
            H=H,
            W=W,
            render_stride=render_stride,
            intrinsics=tgt_K,
            c2w=tgt_c2w,
        )

        ret_dict = {
            "ray_o": view_rays_o.clone(),  # [BxHxW, 3]
            "ray_d": view_rays_d.clone(),  # [BxHxW, 3]
            "camera": data["flat_cam_tgt"],  # [B, 34]
            "rgb": data["rgb_tgt"] if "rgb_tgt" in data else None,  # [B, H, W, 3]
            "batch_refs": batch_refs,  # [BxHxW,]
            "view_uv": view_uvs,  # [BxHxW, 2]
            "raw_h": H,
            "raw_w": W,
            "render_h": render_shape[0],
            "render_w": render_shape[1],
        }

        if isinstance(self.static_renderer, GNTRenderer):
            if render_cfg.gnt_use_masked_spatial_src:
                src_rgbs = data["static_rgb_src_spatial"]
            else:
                src_rgbs = data["rgb_src_spatial"]

            if data["depth_range"].ndim == 4:
                depth_range = data["depth_range"][
                    :, ::render_stride, ::render_stride, :
                ].reshape(
                    (-1, 2)
                )  # [BxHxW, 2]
                depth_range_per_ray = True
            elif data["depth_range"].ndim == 2:
                depth_range = data["depth_range"]  # [B, 2]
                depth_range_per_ray = False
            else:
                raise ValueError(data["depth_range"].shape)

            ret_dict.update(
                {
                    "depth_range": depth_range,  # [BxHxW, 2] or [B, 2]
                    "depth_range_per_ray": depth_range_per_ray,
                    "src_rgbs": src_rgbs,  # [B, #src, H, W, 3]
                    "src_invalid_masks": data[
                        "dyn_mask_src_spatial"
                    ],  # [B, #src, H, W, 1]
                    "src_cameras": data["flat_cam_src_spatial"],  # [B, #src, 34]
                }
            )

            if FLAG_DEBUG_EPIPOLAR:
                ret_dict["debug_pix_coord"] = (
                    260,
                    160,
                )  # (162, 318)  # (160, 447)  # (200, 150)
                ret_dict["debug_dir"] = pathlib.Path(DEBUG_DIR / "epipolar")
                ret_dict["debug_dir"].mkdir(parents=True, exist_ok=True)

                # vis colored point cloud
                if "depth_tgt" in data:
                    mesh_verts = view_rays_o + view_rays_d * data["depth_tgt"].reshape(
                        (-1, 1)
                    )
                    mesh_pcl = trimesh.PointCloud(
                        vertices=mesh_verts.cpu().numpy(),
                        colors=data["rgb_tgt"].reshape((-1, 3)).cpu().numpy(),
                        process=False,
                    )
                    _ = mesh_pcl.export(ret_dict["debug_dir"] / "mesh_pcl.ply")

                # visualize spatial src views
                spatial_K = data["flat_cam_src_spatial"][0, :, 2:18].reshape(
                    (-1, 4, 4)
                )  # [B, #spatial, 34] -> [B, 4, 4]
                spatial_c2w = data["flat_cam_src_spatial"][0, :, 18:34].reshape(
                    (-1, 4, 4)
                )

                b_spatial = spatial_K.shape[0]

                for i in range(b_spatial):
                    (
                        src_spaital_rays_o,
                        src_spatial_rays_d,
                        view_uvs,
                        batch_refs,
                        render_shape,
                    ) = self.get_batched_rays(
                        device=data["rgb_src_spatial"].device,
                        batch_size=1,
                        H=H,
                        W=W,
                        render_stride=1,
                        intrinsics=spatial_K[i : (i + 1), ...],
                        c2w=spatial_c2w[i : (i + 1), ...],
                    )

                    mesh_verts = src_spaital_rays_o + src_spatial_rays_d * data[
                        "depth_src_spatial"
                    ][0, i, ...].reshape((-1, 1))
                    mesh_pcl = trimesh.PointCloud(
                        vertices=mesh_verts.cpu().numpy(),
                        colors=data["rgb_src_spatial"][0, i, ...]
                        .reshape((-1, 3))
                        .cpu()
                        .numpy(),
                        process=False,
                    )
                    _ = mesh_pcl.export(
                        ret_dict["debug_dir"] / f"mesh_pcl_src_spatial_{i:02d}.ply"
                    )

        return ret_dict
