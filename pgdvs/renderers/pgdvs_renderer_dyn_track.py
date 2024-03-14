import logging
import tqdm
import trimesh
import pathlib
import numpy as np

import torch
import torchvision
import pytorch3d.ops as p3d_ops

from pgdvs.renderers.pgdvs_renderer_dyn import PGDVSDynamicRenderer
from pgdvs.models.tapnet.interface import TAPNetInterface
from pgdvs.models.cotracker.interface import CoTrackerInterface


DEBUG_DIR = pathlib.Path(__file__).absolute().parent.parent.parent / "debug/dyn_track"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)


LOGGER = logging.getLogger(__name__)

FLAG_DEBUG_EPIPOLAR = False
LARGE_VAL = 999999


class PGDVSDynamicTrackRenderer(PGDVSDynamicRenderer):
    def render_with_track(
        self, data, render_cfg, base_pcl_info, for_debug=False, disable_tqdm=False
    ):
        device = data["rgb_src_temporal_track_fwd2tgt"].device

        n_b, n_views_one_side, orig_h, orig_w, _ = data[
            "rgb_src_temporal_track_fwd2tgt"
        ].shape

        # last 2 is for two temporally-closest source views
        n_views = n_views_one_side * 2 + 2

        track_rgbs = []
        track_masks = []

        for i_b in range(n_b):
            data_for_track = self.prepare_data(i_b, data, n_views, device)

            dyn_mask_for_track = data_for_track["dyn_masks_for_track"][
                data_for_track["idx_real_track"], ...
            ]  # [#track, H, W, 1]

            if torch.sum(dyn_mask_for_track) > 0:
                # step 1: get all query points from dynamic mask and run tracking
                query_pts, tracks, track_visibles = self.run_track(
                    data_for_track, for_debug=for_debug, disable_tqdm=disable_tqdm
                )

                # step 2: get point cloud
                cur_base_pcl_info = {
                    "pcl": base_pcl_info["pcl"][i_b],
                    "pcl_rgbs": base_pcl_info["pcl_rgbs"][i_b],
                    "pcl_nn_dist_thres": base_pcl_info["pcl_nn_dist_thres"][i_b],
                }
                pcl_track, pcl_track_rgbs = self.compute_pcl_for_tgt(
                    data_for_track=data_for_track,
                    query_pts=query_pts,
                    tracks=tracks,
                    track_visibles=track_visibles,
                    render_cfg=render_cfg,
                    base_pcl_info=cur_base_pcl_info,
                    device=device,
                    for_debug=for_debug,
                )

                # step 3: render with colored point cloud.
                tmp_track_rgb, tmp_track_mask = self.render_dyn_pcl(
                    dyn_mask=torch.zeros_like(
                        data["dyn_mask_src_temporal"][i_b, 0, ...]
                    ),
                    dyn_pcl=pcl_track,
                    rgbs=pcl_track_rgbs,
                    flat_cam=data["flat_cam_tgt"][i_b, ...],
                    render_cfg=render_cfg,
                )
            else:
                tmp_track_rgb = torch.zeros_like(data["rgb_src_temporal"][i_b, 0, ...])
                tmp_track_mask = torch.zeros_like(
                    data["dyn_mask_src_temporal"][i_b, 0, ...]
                )

            track_rgbs.append(tmp_track_rgb)
            track_masks.append(tmp_track_mask)

        track_rgbs = torch.stack(track_rgbs, dim=0).permute(0, 3, 1, 2)  # [B, 3, H, W]
        track_masks = torch.stack(track_masks, dim=0).permute(
            0, 3, 1, 2
        )  # [B, 1, H, W]

        return track_rgbs, track_masks

    def compute_pcl_for_tgt(
        self,
        *,
        data_for_track,
        query_pts,
        tracks,
        track_visibles,
        render_cfg,
        base_pcl_info,
        device,
        for_debug=False,
    ):
        # We need to find query points that are invisible on the two temporally-closest source views.
        # Because if they are visible, then our temporally-closest rendering will have already considered them.
        # For the left query points, we find the frames where they are visible and get corresponding 3D points.
        # We interpolate / extrapolate 3D points to target time step to formulate a point cloud

        # invisible on two temporally closest source views
        vis_temporal_closest = track_visibles[
            :, data_for_track["idx_temporal_closest"]
        ]  # [#pt, 1] or [#pt, 2]
        flag_in_vis_temporal_closest = torch.all(~vis_temporal_closest, dim=1)

        # at least visible on two temporal source views
        vis_real_track = track_visibles[
            :, data_for_track["idx_real_track"]
        ]  # [#pt, #frame]
        flag_vis_enough = torch.sum(vis_real_track.float(), dim=1) >= 2

        flag_valid = flag_in_vis_temporal_closest & flag_vis_enough  # [#pt,]

        n_valid = torch.sum(flag_valid.float()).long()

        if n_valid == 0:
            pcl_track_dummy = torch.zeros((0, 3), device=device)
            pcl_track_rgbs_dummy = torch.zeros((0, 3), device=device)
            return pcl_track_dummy, pcl_track_rgbs_dummy

        valid_query_pts = query_pts[flag_valid, :]  # [#valid, 3]
        valid_tracks = tracks[flag_valid, :, :]  # [#valid, #frame, 2], 2 for [col, row]
        valid_track_visibles = track_visibles[flag_valid, :]  # [#valid, #frame]

        time_for_track = data_for_track["time_for_track"][None, :].expand(
            n_valid, -1
        )  # [#valid, #frame]

        time_tgt = data_for_track["time_tgt"]

        time_diff_to_tgt = (time_for_track - time_tgt).masked_fill_(
            ~valid_track_visibles, float("inf")
        )  # [#valid, #frame]

        # find the two frames whose timesteps are closest to the target timestep
        sorted_frame_idxs = torch.argsort(
            time_diff_to_tgt.abs(), dim=1, descending=False
        )[
            :, :2
        ]  # [#valid, 2]

        # print("\nsorted_frame_idxs: ", sorted_frame_idxs.shape, "\n")

        idx_placeholder = torch.arange(n_valid, device=sorted_frame_idxs.device)[
            :, None
        ].expand(-1, 2)

        time_to_use = time_for_track[idx_placeholder, sorted_frame_idxs][
            :, :2
        ]  # [#valid, 2]
        time_diff_to_use = time_diff_to_tgt[idx_placeholder, sorted_frame_idxs][:, :2]
        assert torch.all(
            ~torch.isinf(time_diff_to_use)
        ), f"{time_diff_to_use.shape}, {torch.sum(torch.isinf(time_diff_to_use))}"

        # print("\ntime_to_use: ", time_to_use.shape, time_diff_to_use.shape, "\n")

        tracks_to_use = torch.cat(
            (
                sorted_frame_idxs[..., None],
                valid_tracks[idx_placeholder, sorted_frame_idxs, :],
            ),
            dim=2,
        )  # [#valid, 2, 3]; 2 for frames; 3 for (frame_idx, col, row)

        flat_tracks_to_use = tracks_to_use.reshape((n_valid * 2, 3))

        unique_frame_idxs = torch.unique(flat_tracks_to_use[:, 0]).long().tolist()

        flat_track_idxs = torch.arange(flat_tracks_to_use.shape[0]).to(device)
        shuffled_flat_track_idxs = []

        flat_pt_coords_3d = []
        flat_pt_rgbs = []

        for frame_i in unique_frame_idxs:
            unique_mask = flat_tracks_to_use[:, 0] == frame_i
            unique_uvs = flat_tracks_to_use[unique_mask, 1:]  # [#unique, 2], (col, row)

            shuffled_flat_track_idxs.append(flat_track_idxs[unique_mask])

            frame_rgb = data_for_track["rgbs_for_track"][
                frame_i : (frame_i + 1), ...
            ].permute(
                0, 3, 1, 2
            )  # [1, 1, H, W]

            _, _, tmp_h, tmp_w = frame_rgb.shape
            unique_grid_uvs = (
                2 * unique_uvs / torch.FloatTensor((tmp_w, tmp_h))[None, :].to(device)
                - 1
            )  # range [-1, 1]

            unique_rgbs = torch.nn.functional.grid_sample(
                frame_rgb,
                unique_grid_uvs[None, None, ...],
                mode="bilinear",
                align_corners=True,
            )[0, :, 0, :].permute(
                1, 0
            )  # [#unique, 3]

            flat_pt_rgbs.append(unique_rgbs)

            frame_depth = data_for_track["depths_for_track"][
                frame_i : (frame_i + 1), ...
            ].permute(
                0, 3, 1, 2
            )  # [1, 1, H, W]
            unique_depths = torch.nn.functional.grid_sample(
                frame_depth, unique_grid_uvs[None, None, ...], mode="nearest"
            )[
                0, 0, 0, :
            ]  # [#unique, ]

            frame_flat_cam = data_for_track["flat_cams_for_track"][frame_i, :]  # [34,]
            frame_K = frame_flat_cam[2:18].reshape((4, 4))
            frame_c2w = frame_flat_cam[18:34].reshape((4, 4))

            unique_uvs_homo = torch.nn.functional.pad(
                unique_uvs, (0, 1), value=1
            ).permute(1, 0)

            unique_rays_d = (
                frame_c2w[None, :3, :3]
                .bmm(torch.inverse(frame_K[None, :3, :3]))
                .bmm(unique_uvs_homo[None, ...])
            ).transpose(1, 2)[
                0, ...
            ]  # [1, 3, 3] x [1, 3, 3] x [1, 3, #unique] -> [1, 3, #unique] -> [#unique, 3]
            unique_rays_o = frame_c2w[None, :3, 3].repeat(
                unique_rays_d.shape[0], 1
            )  # [#unique, 3]

            unique_coords_3d = (
                unique_rays_o + unique_rays_d * unique_depths[:, None]
            )  # [#unique, 3]
            flat_pt_coords_3d.append(unique_coords_3d)

            if for_debug:
                mesh_pcl = trimesh.PointCloud(
                    vertices=unique_coords_3d.cpu().numpy(),
                    colors=unique_rgbs.cpu().numpy(),
                    process=False,
                )
                _ = mesh_pcl.export(
                    DEBUG_DIR / f"mesh_pcl_track_progress_{frame_i:02d}.ply"
                )

        sorted_flat_track_idxs = torch.argsort(
            torch.cat(shuffled_flat_track_idxs, dim=0)
        )
        pt_coords_3d = torch.cat(flat_pt_coords_3d, dim=0)[
            sorted_flat_track_idxs, ...
        ].reshape((n_valid, 2, 3))
        pcl_track_rgbs = torch.mean(
            torch.cat(flat_pt_rgbs, dim=0)[sorted_flat_track_idxs, ...].reshape(
                (n_valid, 2, 3)
            ),
            dim=1,
        )  # [#valid, 3]

        ratio = (time_tgt - time_to_use[:, :1]) / (
            time_to_use[:, 1:2] - time_to_use[:, :1] + 1e-8
        )
        pcl_track = (
            pt_coords_3d[:, 0, :]
            + (pt_coords_3d[:, 1, :] - pt_coords_3d[:, 0, :]) * ratio
        )  # [#valid, 3]

        if pcl_track.shape[0] > 0:
            if for_debug:
                mesh_pcl = trimesh.PointCloud(
                    vertices=pcl_track.cpu().numpy(),
                    colors=pcl_track_rgbs.cpu().numpy(),
                    process=False,
                )
                _ = mesh_pcl.export(DEBUG_DIR / "mesh_pcl_track.ply")

            # remove points that are too far away from base pcl
            if (base_pcl_info["pcl"] is not None) and (
                base_pcl_info["pcl"].shape[0] > 0
            ):
                (
                    nn_dists_track2base,
                    nn_idxs_track2base,
                    nn_pts_track2base,
                ) = p3d_ops.knn_points(
                    pcl_track[None, ...],
                    base_pcl_info["pcl"][None, ...],
                    K=(render_cfg.dyn_pcl_outlier_knn + 1),
                    return_nn=True,
                )  # nn_dists/idxs: [1, #valid, K]; nn_pts: [1, #valid, K, 3]

                avg_nn_dist_track2base = torch.mean(
                    nn_dists_track2base[0, ...], dim=1
                )  # [#pt, ]

                flag_not_outlier_track2base = (
                    avg_nn_dist_track2base
                    < base_pcl_info["pcl_nn_dist_thres"]
                    * render_cfg.dyn_pcl_track_track2base_thres_mult
                )

                assert (
                    flag_not_outlier_track2base.shape[0] == pcl_track.shape[0]
                ), f"{flag_not_outlier_track2base.shape}, {pcl_track.shape}"
                assert (
                    flag_not_outlier_track2base.shape[0] == pcl_track_rgbs.shape[0]
                ), f"{flag_not_outlier_track2base.shape}, {pcl_track_rgbs.shape}"

                pcl_track = pcl_track[flag_not_outlier_track2base, :]
                pcl_track_rgbs = pcl_track_rgbs[flag_not_outlier_track2base, :]

                if for_debug and (pcl_track.shape[0] > 0):
                    mesh_pcl = trimesh.PointCloud(
                        vertices=pcl_track.cpu().numpy(),
                        colors=pcl_track_rgbs.cpu().numpy(),
                        process=False,
                    )
                    _ = mesh_pcl.export(
                        DEBUG_DIR / "mesh_pcl_track_track2base_clean.ply"
                    )

            if pcl_track.shape[0] > 0:
                # removal of outliers
                # - https://github.com/facebookresearch/pytorch3d/issues/511#issuecomment-1152970392
                # - http://www.open3d.org/docs/release/tutorial/geometry/pointcloud_outlier_removal.html
                # - https://pcl.readthedocs.io/en/latest/statistical_outlier.html
                nn_dists, nn_idxs, nn_pts = p3d_ops.knn_points(
                    pcl_track[None, ...],
                    pcl_track[None, ...],
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

                if base_pcl_info["pcl_nn_dist_thres"] is not None:
                    nn_dist_thres = base_pcl_info["pcl_nn_dist_thres"]
                else:
                    nn_dist_thres = (
                        dyn_pcl_nn_dist_med
                        + dyn_pcl_nn_dist_std * render_cfg.dyn_pcl_outlier_std_thres
                    )

                flag_not_outlier = avg_nn_dist < nn_dist_thres
                assert (
                    flag_not_outlier.shape[0] == pcl_track.shape[0]
                ), f"{flag_not_outlier.shape}, {pcl_track.shape}"
                assert (
                    flag_not_outlier.shape[0] == pcl_track_rgbs.shape[0]
                ), f"{flag_not_outlier.shape}, {pcl_track_rgbs.shape}"

                pcl_track = pcl_track[flag_not_outlier, :]
                pcl_track_rgbs = pcl_track_rgbs[flag_not_outlier, :]

                if for_debug and (pcl_track.shape[0] > 0):
                    mesh_pcl = trimesh.PointCloud(
                        vertices=pcl_track.cpu().numpy(),
                        colors=pcl_track_rgbs.cpu().numpy(),
                        process=False,
                    )
                    _ = mesh_pcl.export(DEBUG_DIR / "mesh_pcl_track_clean.ply")

            if (base_pcl_info["pcl"] is not None) and (pcl_track.shape[0] > 0):
                pcl_track = torch.cat((pcl_track, base_pcl_info["pcl"]), dim=0)
                pcl_track_rgbs = torch.cat(
                    (pcl_track_rgbs, base_pcl_info["pcl_rgbs"]), dim=0
                )

        return pcl_track, pcl_track_rgbs

    def run_track(self, data_for_track, for_debug=False, disable_tqdm=False):
        if for_debug:
            grid_img_in_tensor = torchvision.utils.make_grid(
                data_for_track["dyn_masks_for_track"].permute(0, 3, 1, 2),
                nrow=4,
                value_range=[0, 1],
                padding=5,
                pad_value=1.0,
            )  # [3, H, W]
            torchvision.utils.save_image(
                grid_img_in_tensor, DEBUG_DIR / "dyn_track_in_mask.png"
            )

        if isinstance(self.tracker, TAPNetInterface):
            query_pts, tracks, visibles = self.run_track_func(
                data_for_track,
                track_k="idx_real_track",
                track_bwd=False,
                for_debug=for_debug,
                disable_tqdm=disable_tqdm,
            )
        elif isinstance(self.tracker, CoTrackerInterface):
            # separate forward and backward
            query_pts = []
            tracks = []
            visibles = []

            if len(data_for_track["idx_real_track_fwd"]) > 0:
                dyn_mask_for_track_fwd = data_for_track["dyn_masks_for_track"][
                    data_for_track["idx_real_track_fwd"], ...
                ]  # [#track, H, W, 1]
                if torch.sum(dyn_mask_for_track_fwd) > 0:
                    query_pts_fwd, tracks_fwd, visibles_fwd = self.run_track_func(
                        data_for_track,
                        track_k="idx_real_track_fwd",
                        track_bwd=False,
                        for_debug=for_debug,
                        disable_tqdm=disable_tqdm,
                    )
                    query_pts.append(query_pts_fwd)
                    tracks.append(tracks_fwd)
                    visibles.append(visibles_fwd)

            if len(data_for_track["idx_real_track_bwd"]) > 0:
                dyn_mask_for_track_bwd = data_for_track["dyn_masks_for_track"][
                    data_for_track["idx_real_track_bwd"], ...
                ]  # [#track, H, W, 1]
                if torch.sum(dyn_mask_for_track_bwd) > 0:
                    query_pts_bwd, tracks_bwd, visibles_bwd = self.run_track_func(
                        data_for_track,
                        track_k="idx_real_track_bwd",
                        track_bwd=True,
                        for_debug=for_debug,
                        disable_tqdm=disable_tqdm,
                    )
                    query_pts.append(query_pts_bwd)
                    tracks.append(tracks_bwd)
                    visibles.append(visibles_bwd)

            query_pts = torch.cat(query_pts, dim=0)
            tracks = torch.cat(tracks, dim=0)
            visibles = torch.cat(visibles, dim=0)
        else:
            raise TypeError(type(self.tracker))

        if for_debug:
            self.debug_vis_track(
                data_for_track, tracks, visibles, save_name="dyn_track_traj"
            )
            # import sys
            # sys.exit(1)

        return query_pts, tracks, visibles

    def run_track_func(
        self,
        data_for_track,
        track_k="idx_real_track",
        track_bwd=False,
        for_debug=False,
        disable_tqdm=False,
    ):
        query_pts = []
        for idx in data_for_track[track_k]:
            tmp_rows, tmp_cols = torch.nonzero(
                data_for_track["dyn_masks_for_track"][idx, ..., 0] > 0.0, as_tuple=True
            )
            tmp_time = torch.ones_like(tmp_rows) * idx
            # see sample_random_points of https://github.com/deepmind/tapnet/blob/4ac6b2acd0aed36c0762f4247de9e8630340e2e0/colabs/tapir_demo.ipynb
            query_pts.append(torch.stack((tmp_time, tmp_rows, tmp_cols), dim=1).float())
        query_pts = torch.cat(query_pts, dim=0)  # [#pt, 3]

        n_actual_pts = query_pts.shape[0]

        # To make Jax run fast we need have fixed-shape tensor:
        # https://github.com/deepmind/tapnet/issues/42#issuecomment-1664506310
        n_final_pts = int(
            np.ceil(n_actual_pts / self.track_chunk_size) * self.track_chunk_size
        )
        n_padded_pts = n_final_pts - n_actual_pts
        query_pts = torch.cat((query_pts, query_pts[:n_padded_pts, ...]), dim=0)

        n_actual_frames = data_for_track["n_actual_frames"]
        n_frames = data_for_track["rgbs_for_track"].shape[0]
        if track_bwd:
            rgbs_for_track = torch.flip(
                data_for_track["rgbs_for_track"][:n_actual_frames, ...], dims=[0]
            )
            if n_actual_frames != n_frames:
                n_pad_frames = n_frames - n_actual_frames
                rgbs_for_track = torch.cat(
                    (rgbs_for_track, rgbs_for_track[-n_pad_frames:, ...]), dim=0
                )
            query_pts[:, 0] = n_actual_frames - 1 - query_pts[:, 0]
            assert torch.all(query_pts[:, 0] >= 0), f"{torch.sum(query_pts[:, 0] >= 0)}"
        else:
            rgbs_for_track = data_for_track["rgbs_for_track"]

        tracks = []
        visibles = []

        for tmp_start in tqdm.tqdm(
            torch.arange(0, n_final_pts, self.track_chunk_size),
            disable=disable_tqdm,
            desc="dyn_track",
        ):
            tmp_end = min(tmp_start + self.track_chunk_size, n_final_pts)
            tmp_query_pts = query_pts[tmp_start:tmp_end, :]
            tmp_tracks, tmp_visibles = self.tracker(
                frames=rgbs_for_track, query_points=tmp_query_pts
            )

            if track_bwd:
                tmp_tracks = torch.flip(tmp_tracks[:, :n_actual_frames, :], dims=[1])
                tmp_visibles = torch.flip(tmp_visibles[:, :n_actual_frames], dims=[1])

            tracks.append(tmp_tracks)
            visibles.append(tmp_visibles)

            if for_debug:
                self.debug_vis_track(
                    data_for_track,
                    tmp_tracks,
                    tmp_visibles,
                    save_name=f"dyn_track_traj_{track_k}",
                )
                # import sys
                # sys.exit(1)

        # https://github.com/deepmind/tapnet/blob/4ac6b2acd0aed36c0762f4247de9e8630340e2e0/utils/viz_utils.py#L96-L98
        tracks = torch.cat(tracks, dim=0)[
            :n_actual_pts, :n_actual_frames, :
        ]  # tracks: [#pt, #frame, 2], float32, 2 for [col, row] or [u, v]
        visibles = torch.cat(visibles, dim=0)[
            :n_actual_pts, :n_actual_frames
        ]  # visibles: [#pt, #frame], bool

        if track_bwd:
            query_pts[:, 0] = n_actual_frames - 1 - query_pts[:, 0]

        return query_pts[:n_actual_pts, ...], tracks, visibles

    def debug_vis_track(
        self, data_for_track, tracks, visibles, save_name="dyn_track_traj"
    ):
        n_actual_frames = data_for_track["n_actual_frames"]
        debug_rgbs = data_for_track["rgbs_for_track"][
            :n_actual_frames, ...
        ].clone()  # [N, H, W, 3]
        tmp_n, tmp_h, tmp_w, _ = debug_rgbs.shape
        debug_idx = -1000
        debug_traj = tracks[debug_idx, ...]  # [#actual, 2]
        debug_vis = visibles[debug_idx, :]
        tmp_range = 5
        tmp_color_vis = torch.zeros((1, 1, 3), device=debug_rgbs.device)
        tmp_color_vis[0, 0, 0] = 1.0  # red
        tmp_color_invis = torch.zeros((1, 1, 3), device=debug_rgbs.device)
        tmp_color_invis[0, 0, 2] = 1.0  # green
        for tmp_i in range(tmp_n):
            tmp_col = debug_traj[tmp_i, 0].long()
            tmp_row = debug_traj[tmp_i, 1].long()
            tmp_vis = debug_vis[tmp_i]
            if tmp_vis:
                tmp_color = tmp_color_vis
            else:
                tmp_color = tmp_color_invis
            debug_rgbs[
                tmp_i,
                max(0, tmp_row - tmp_range) : min(tmp_h, tmp_row + tmp_range),
                max(0, tmp_col - tmp_range) : min(tmp_w, tmp_col + tmp_range),
                :,
            ] = tmp_color
        grid_img_in_tensor = torchvision.utils.make_grid(
            debug_rgbs.permute(0, 3, 1, 2),
            nrow=4,
            value_range=[0, 1],
            padding=5,
            pad_value=1.0,
        )  # [3, H, W]
        torchvision.utils.save_image(grid_img_in_tensor, DEBUG_DIR / f"{save_name}.png")

    def prepare_data(self, i_b, data, n_views, device):
        rgbs_for_track = []
        dyn_masks_for_track = []
        depths_for_track = []
        flat_cams_for_track = []
        time_for_track = []

        idx_temporal_closest = []
        idx_real_track = []
        idx_real_track_fwd = []
        idx_real_track_bwd = []

        n_actual_frames = 0

        n_actual_temporal_track_fwd2tgt = data["n_actual_temporal_track_fwd2tgt"][
            i_b, 0
        ]
        if n_actual_temporal_track_fwd2tgt > 0:
            rgbs_for_track.append(
                data["rgb_src_temporal_track_fwd2tgt"][
                    i_b, :n_actual_temporal_track_fwd2tgt
                ]
            )
            dyn_masks_for_track.append(
                data["dyn_mask_src_temporal_track_fwd2tgt"][
                    i_b, :n_actual_temporal_track_fwd2tgt
                ]
            )
            depths_for_track.append(
                data["depth_src_temporal_track_fwd2tgt"][
                    i_b, :n_actual_temporal_track_fwd2tgt
                ]
            )
            flat_cams_for_track.append(
                data["flat_cam_src_temporal_track_fwd2tgt"][
                    i_b, :n_actual_temporal_track_fwd2tgt
                ]
            )
            time_for_track.append(
                data["time_src_temporal_track_fwd2tgt"][
                    i_b, :n_actual_temporal_track_fwd2tgt
                ]
            )

            idx_real_track_fwd = torch.arange(
                n_actual_temporal_track_fwd2tgt, device=device
            ).tolist()
            idx_real_track.extend(idx_real_track_fwd)
            n_actual_frames = n_actual_frames + n_actual_temporal_track_fwd2tgt

        n_actual_temporal = data["n_actual_temporal"][i_b, 0]
        idx_temporal_closest = (
            torch.arange(n_actual_temporal, device=device) + n_actual_frames
        ).tolist()

        rgbs_for_track.append(
            data["rgb_src_temporal"][i_b, :n_actual_temporal, ...]
        )  # [1, H, W, 3] or [2, H, W, 3]
        depths_for_track.append(data["depth_src_temporal"][i_b, :n_actual_temporal])
        flat_cams_for_track.append(
            data["flat_cam_src_temporal"][i_b, :n_actual_temporal]
        )
        dyn_masks_for_track.append(
            data["dyn_mask_src_temporal"][i_b, :n_actual_temporal]
        )

        time_temporal_closest = data["time_src_temporal"][i_b, :n_actual_temporal]
        time_for_track.append(time_temporal_closest)
        n_actual_frames = n_actual_frames + n_actual_temporal

        n_actual_temporal_track_bwd2tgt = data["n_actual_temporal_track_bwd2tgt"][
            i_b, 0
        ]
        if n_actual_temporal_track_bwd2tgt > 0:
            rgbs_for_track.append(
                data["rgb_src_temporal_track_bwd2tgt"][
                    i_b, :n_actual_temporal_track_bwd2tgt
                ]
            )
            dyn_masks_for_track.append(
                data["dyn_mask_src_temporal_track_bwd2tgt"][
                    i_b, :n_actual_temporal_track_bwd2tgt
                ]
            )
            depths_for_track.append(
                data["depth_src_temporal_track_bwd2tgt"][
                    i_b, :n_actual_temporal_track_bwd2tgt
                ]
            )
            flat_cams_for_track.append(
                data["flat_cam_src_temporal_track_bwd2tgt"][
                    i_b, :n_actual_temporal_track_bwd2tgt
                ]
            )
            time_for_track.append(
                data["time_src_temporal_track_bwd2tgt"][
                    i_b, :n_actual_temporal_track_bwd2tgt
                ]
            )

            idx_real_track_bwd = (
                torch.arange(n_actual_temporal_track_bwd2tgt, device=device)
                + n_actual_frames
            ).tolist()
            idx_real_track.extend(idx_real_track_bwd)

        # print("\nidx_real_track: ", idx_real_track, idx_temporal_closest, "\n")

        idx_full = idx_temporal_closest + idx_real_track
        assert len(idx_full) == len(set(idx_full)), f"{idx_full}"
        diff_set = set(idx_full).difference(set(np.arange(len(idx_full)).tolist()))
        assert len(diff_set) == 0, f"{diff_set}"

        rgbs_for_track = torch.cat(rgbs_for_track, dim=0)  # [N, H, W, 3]
        dyn_masks_for_track = torch.cat(dyn_masks_for_track, dim=0)  # [N, H, W, 1]
        depths_for_track = torch.cat(depths_for_track, dim=0)  # [N, H, W, 1]
        flat_cams_for_track = torch.cat(flat_cams_for_track, dim=0)  # [N, 34]
        time_for_track = torch.cat(time_for_track, dim=0)

        min_time = torch.min(time_for_track)
        time_for_track = time_for_track - min_time  # normalize to start from 0

        time_tgt = data["time_tgt"][i_b, :] - min_time

        # To make Jax run fast we need have fixed-shape tensor:
        # https://github.com/deepmind/tapnet/issues/42#issuecomment-1664506310
        n_actual_frames = rgbs_for_track.shape[0]
        n_rep = int(np.ceil(n_views / n_actual_frames))

        rgbs_for_track_padded = rgbs_for_track.repeat(n_rep, 1, 1, 1)[:n_views, ...]
        dyn_masks_for_track_padded = dyn_masks_for_track.repeat(n_rep, 1, 1, 1)[
            :n_views, ...
        ]

        assert (
            rgbs_for_track_padded.shape[0] == n_views
        ), f"{rgbs_for_track_padded.shape}, {n_views}"
        assert (
            dyn_masks_for_track_padded.shape[0] == n_views
        ), f"{dyn_masks_for_track_padded.shape}, {n_views}"
        assert (
            depths_for_track.shape[0] == n_actual_frames
        ), f"{depths_for_track.shape}, {n_actual_frames}"
        assert (
            flat_cams_for_track.shape[0] == n_actual_frames
        ), f"{flat_cams_for_track.shape}, {n_actual_frames}"
        assert (
            time_for_track.shape[0] == n_actual_frames
        ), f"{time_for_track.shape}, {n_actual_frames}"

        ret_dict = {
            "n_actual_frames": n_actual_frames,
            "rgbs_for_track": rgbs_for_track_padded,
            "dyn_masks_for_track": dyn_masks_for_track_padded,
            "depths_for_track": depths_for_track,
            "flat_cams_for_track": flat_cams_for_track,
            "time_for_track": time_for_track,  # [N, ]
            "time_tgt": time_tgt,  # [1, ]
            "idx_temporal_closest": idx_temporal_closest,  # [1, ] or [2, ]
            "idx_real_track": idx_real_track,
            "idx_real_track_fwd": idx_real_track_fwd,
            "idx_real_track_bwd": idx_real_track_bwd,
            "time_real_track": time_for_track[idx_real_track],
        }

        return ret_dict
