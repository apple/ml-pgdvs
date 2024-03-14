import os
import tqdm
import pathlib
import hydra
import hydra.utils
import numpy as np
from collections import OrderedDict

import torch
import pytorch3d
import pytorch3d.utils
import pytorch3d.ops as p3d_ops

from pgdvs.utils.training import disabled_train
from pgdvs.utils.vis_utils import draw_cam_mesh
from pgdvs.models.gnt.common import TINY_NUMBER, HUGE_NUMBER
from pgdvs.models.gnt.projector import Projector


class StaticGeoPointRenderer(torch.nn.Module):
    def __init__(self, model_cfg=None):
        super().__init__()

        self.projector = Projector()

    def forward(self, *, tgt_h, tgt_w, flat_tgt_cam, st_pcl_rgb, render_cfg):
        assert st_pcl_rgb.ndim == 2, f"{st_pcl_rgb.shape}"

        st_pcl = st_pcl_rgb[:, :3]  # [#pt, 3]
        st_rgb = st_pcl_rgb[:, 3:]  # [#pt, 3]

        if render_cfg.st_pcl_remove_outlier:
            # removal of outliers
            # - https://github.com/facebookresearch/pytorch3d/issues/511#issuecomment-1152970392
            # - http://www.open3d.org/docs/release/tutorial/geometry/pointcloud_outlier_removal.html
            # - https://pcl.readthedocs.io/en/latest/statistical_outlier.html
            nn_dists, nn_idxs, nn_pts = p3d_ops.knn_points(
                st_pcl[None, ...],
                st_pcl[None, ...],
                K=(render_cfg.st_pcl_outlier_knn + 1),
                return_nn=True,
            )  # nn_dists/idxs: [1, #pts, K]; nn_pts: [1, #pts, K, 3]

            # The 1st distance is always 0 as the nearest is the point itself.
            nn_dists = nn_dists[0, :, 1:]
            nn_idxs = nn_idxs[0, :, 1:]
            nn_pts = nn_pts[0, :, 1:, :]

            # We mimic Open3D's statistical removal
            # https://github.com/isl-org/Open3D/blob/6ddbcd5c9b8bf0b496e4151c7d7766af09e3dba7/cpp/open3d/geometry/PointCloud.cpp#L636-L653
            avg_nn_dist = torch.mean(nn_dists, dim=1)  # [#pt, ]
            st_pcl_nn_dist_med = torch.median(avg_nn_dist)
            st_pcl_nn_dist_std = torch.std(avg_nn_dist)
            nn_dist_thres = (
                st_pcl_nn_dist_med
                + st_pcl_nn_dist_std * render_cfg.st_pcl_outlier_std_thres
            )

            flag_not_outlier = avg_nn_dist < nn_dist_thres
            assert (
                flag_not_outlier.shape[0] == st_pcl.shape[0]
            ), f"{flag_not_outlier.shape}, {st_pcl.shape}"

            st_pcl_clean = st_pcl[flag_not_outlier, :]
        else:
            st_pcl_clean = st_pcl
            flag_not_outlier = torch.ones(
                (st_pcl.shape[0]), dtype=bool, device=st_pcl.device
            )

        assert (
            flag_not_outlier.shape[0] == st_rgb.shape[0]
        ), f"{flag_not_outlier.shape}, {st_rgb.shape}"

        st_rgb_clean = st_rgb[flag_not_outlier, :]

        K = flat_tgt_cam[2:18].reshape((4, 4))
        c2w = flat_tgt_cam[18:34].reshape((4, 4))
        w2c = torch.inverse(c2w)

        if st_pcl_clean.shape[0] == 0:
            mesh_img = torch.zeros((tgt_h, tgt_w, 3))  # [H, W, 3]
            mesh_mask = torch.zeros((tgt_h, tgt_w, 1))  # [H, W, 1]
        else:
            img_size = torch.LongTensor([tgt_h, tgt_w]).reshape((1, 2))
            cameras_pytorch3d = pytorch3d.utils.cameras_from_opencv_projection(
                w2c[None, :3, :3], w2c[None, :3, 3], K[None, :3, :3], img_size
            )

            # for bin size, see https://github.com/facebookresearch/pytorch3d/issues/1064
            raster_settings = pytorch3d.renderer.PointsRasterizationSettings(
                image_size=(tgt_h, tgt_w),
                radius=render_cfg.st_render_pcl_pt_radius,
                points_per_pixel=render_cfg.st_render_pcl_pts_per_pixel,
                bin_size=0,
            )

            # Create a points renderer by compositing points using an alpha compositor (nearer points
            # are weighted more heavily). See [1] for an explanation.
            rasterizer = pytorch3d.renderer.PointsRasterizer(
                cameras=cameras_pytorch3d, raster_settings=raster_settings
            )
            point_renderer = pytorch3d.renderer.PointsRenderer(
                rasterizer=rasterizer,
                compositor=pytorch3d.renderer.NormWeightedCompositor(
                    background_color=(0, 0, 0)
                ),
            )

            dy_mesh = pytorch3d.structures.Pointclouds(
                points=st_pcl_clean[None, ...],  # [1, #pts, 3]
                features=st_rgb_clean[None, ...],  # [1, #pt, 3]
            )

            mesh_img = point_renderer(dy_mesh)[0, :, :, :3]  # [H, W, 3]

            dy_mesh.features = torch.ones_like(st_rgb_clean)[None, ...]
            mesh_mask = (
                point_renderer(dy_mesh)[0, :, :, :1] > 0.0
            ).float()  # [H, W, 1]

        return mesh_img, mesh_mask
