import cv2
import copy
import tqdm
import trimesh
import logging
import pathlib
import trimesh
import PIL.Image
import numpy as np

import torch

from pgdvs.utils.vis_utils import draw_cam_mesh
from pgdvs.utils.rendering import modify_rgb_range
from pgdvs.datasets.base import BaseDataset
from pgdvs.datasets.nvidia_vis import draw_set_poses, create_bt_poses
from pgdvs.utils.geometry import linear_pose_interp, recenter_poses


DEBUG_DIR = pathlib.Path(__file__).absolute().parent.parent.parent / "debug/mono_vis"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)

FLAG_RESCALE_POSES = False
FLAG_CENTER_POSES = False

FLAG_SAVE_CAM_MESH = False

N_BT_REPS = 8


LOGGER = logging.getLogger(__name__)


class MonoVisualizationDataset(BaseDataset):
    dataset_name = "Monocular Visualization"
    dataset_fname = "mono_vis"

    def __init__(
        self,
        *,
        data_root,
        max_hw,
        mode,
        rgb_range="0_1",
        use_aug=False,
        scene_ids=None,
        n_src_views_spatial=10,
        n_src_views_temporal_track_one_side=5,
        # vis
        vis_center_time=50,
        n_render_frames=200,
        vis_time_interval=10,
        vis_bt_max_disp=32,
        flow_consist_thres=1.0,
    ):
        self.mode = mode

        assert (
            max_hw == -1
        ), f"We enforce to use raw resolution. However, we receive max_hw of {max_hw}"
        self.max_hw = max_hw

        assert not use_aug
        self.use_aug = use_aug

        self.use_normalized_K = False

        self.rgb_range = rgb_range
        self.n_src_views_spatial = n_src_views_spatial
        self.n_src_views_temporal_track_one_side = n_src_views_temporal_track_one_side

        self.flow_consist_thres = flow_consist_thres

        self.data_root = pathlib.Path(data_root)
        assert self.data_root.exists(), self.data_root

        LOGGER.info(
            f"[{self.dataset_name}] [{self.mode}] data_root: {self.data_root}.\n"
        )

        assert self.mode in ["vis"], f"{self.mode}"

        LOGGER.info(f"[{self.dataset_name}] [{self.mode}] use_aug: {use_aug}\n")

        LOGGER.info(f"[{self.dataset_name}] [{self.mode}]\n")

        assert scene_ids is not None

        LOGGER.info(
            f"[{self.dataset_name}] [{self.mode}] scene_ids: {scene_ids}, {type(scene_ids)}\n"
        )

        c2w_dict = {}
        K_dict = {}
        all_vis_poses = []

        for tmp_scene_id in scene_ids:
            tmp_scene_dir = self.data_root / tmp_scene_id

            all_K = []
            all_c2w = []

            tmp_pose_f_list = sorted(list((tmp_scene_dir / "poses").glob("*.npz")))
            for tmp_f in tmp_pose_f_list:
                cam_info = np.load(tmp_f)
                all_K.append(cam_info["K"])
                all_c2w.append(cam_info["c2w"])

            all_K = np.array(all_K)  # [#frame, 4 4]
            all_c2w = np.array(all_c2w)  # [#frame, 4 4]

            n_img_fs = all_K.shape[0]

            render_time_list = np.linspace(
                max(0, vis_center_time - vis_time_interval),
                min(n_img_fs - 2, vis_center_time + vis_time_interval),
                n_render_frames,
            ).tolist()

            if FLAG_SAVE_CAM_MESH:
                draw_set_poses(
                    all_c2w,
                    DEBUG_DIR / "vis_poses_raw.ply",
                    tmp_coord=1.0,
                )

            tmp_depth_f_list = sorted(list((tmp_scene_dir / "depths").glob("*.npz")))

            all_bounds = []
            for tmp_f in tmp_depth_f_list:
                zs = np.load(tmp_f)["depth"].reshape((-1))
                close_depth, inf_depth = np.percentile(zs, 5), np.percentile(zs, 95)
                bounds = np.array([close_depth, inf_depth])
                all_bounds.append(bounds)
            all_bounds = np.array(all_bounds)

            bd_factor = 0.9
            if FLAG_RESCALE_POSES:
                # TODO: do we need this though: https://github.com/zhengqili/Neural-Scene-Flow-Fields/blob/d4001759a39b056c95d8bc22da34b10b4fb85afb/nsff_exp/load_llff.py#L263-L271
                # Rescale if bd_factor is provided
                # - https://github.com/bmild/nerf/issues/34
                # - Appendix C in https://arxiv.org/pdf/2003.08934.pdf
                pose_sc = 1.0 / (np.percentile(all_bounds[:, 0], 5) * bd_factor)
                bt_disp_sc = 1.0
            else:
                pose_sc = 1.0
                bt_disp_sc = 1.0 / (np.percentile(all_bounds[:, 0], 5) * bd_factor)

            all_c2w[:, :3, 3] *= pose_sc  # camera position

            if FLAG_SAVE_CAM_MESH:
                draw_set_poses(
                    all_c2w,
                    DEBUG_DIR / "vis_poses_sc.ply",
                    tmp_coord=1.0,
                )

            if FLAG_CENTER_POSES:
                all_c2w = recenter_poses(all_c2w)  # [#frame, 3, 5]

                if FLAG_SAVE_CAM_MESH:
                    draw_set_poses(
                        all_c2w,
                        DEBUG_DIR / "vis_poses_centered.ply",
                        tmp_coord=1.0,
                    )

            bt_poses = create_bt_poses(
                all_K[0, 0, 0],
                num_frames=len(render_time_list) // N_BT_REPS,
                sc=bt_disp_sc,
                max_disp=vis_bt_max_disp,
            )
            bt_poses = bt_poses * (N_BT_REPS + 1)  # * int(10 * n_frames / 24)

            c2w_dict[tmp_scene_id] = np.copy(all_c2w)
            K_dict[tmp_scene_id] = np.copy(all_K)

            if FLAG_SAVE_CAM_MESH:
                all_verts = []
                all_verts_2 = []
                all_colors = []

            for cur_i, cur_time in enumerate(tqdm.tqdm(render_time_list)):
                int_time = int(np.floor(cur_time))
                ratio = cur_time - np.floor(cur_time)

                interp_rot, interp_trans = linear_pose_interp(
                    all_c2w[int_time, :3, 3],
                    all_c2w[int_time, :3, :3],
                    all_c2w[int_time + 1, :3, 3],
                    all_c2w[int_time + 1, :3, :3],
                    ratio,
                )

                interp_c2w = np.concatenate(
                    (interp_rot, interp_trans[:, np.newaxis]), 1
                )  # [3, 4]
                interp_c2w = np.concatenate(
                    [
                        interp_c2w[:3, :4],
                        np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :],
                    ],
                    axis=0,
                )  # [4, 4]

                interp_c2w = np.dot(interp_c2w, bt_poses[cur_i])  # [4, 4]

                tmp_elem = [
                    tmp_scene_id,
                    tmp_scene_dir,
                    cur_time,
                    cur_i,
                    interp_c2w,
                    pose_sc,
                ]
                all_vis_poses.append(tmp_elem)

                if FLAG_SAVE_CAM_MESH:
                    tmp_verts, tmp_colors = draw_cam_mesh(
                        np.linalg.inv(interp_c2w),
                        None,
                        tmp_coord=0.5,
                        flag_save=False,
                    )
                    tmp_verts_2, _ = draw_cam_mesh(
                        np.linalg.inv(bt_poses[cur_i]),
                        None,
                        tmp_coord=0.5 * 0.1 * (vis_bt_max_disp / 256),
                        flag_save=False,
                    )
                    tmp_colors = (
                        tmp_colors.astype(float) * (cur_i + 1) / (n_render_frames + 2)
                    ).astype(np.uint8)
                    all_verts.append(tmp_verts)
                    all_verts_2.append(tmp_verts_2)
                    all_colors.append(tmp_colors)

            if FLAG_SAVE_CAM_MESH:
                all_verts = np.concatenate(all_verts, axis=0)
                all_verts_2 = np.concatenate(all_verts_2, axis=0)
                all_colors = np.concatenate(all_colors, axis=0)

                cam_mesh = trimesh.points.PointCloud(
                    vertices=all_verts, colors=all_colors
                )
                _ = cam_mesh.export(DEBUG_DIR / "vis_poses_bt.ply")

                cam_mesh_2 = trimesh.points.PointCloud(
                    vertices=all_verts_2, colors=all_colors
                )
                _ = cam_mesh_2.export(DEBUG_DIR / "vis_poses_bt_supp.ply")

        self.c2w_dict = c2w_dict
        self.K_dict = K_dict

        self.valid_fs = all_vis_poses

    def __len__(self):
        return len(self.valid_fs)

    def __getitem__(self, index):
        scene_id, scene_dir, tgt_time, tgt_idx, tgt_c2w, c2w_sc = self.valid_fs[index]

        rgb_dir = pathlib.Path(scene_dir) / "rgbs"
        img_exts = PIL.Image.registered_extensions()
        supported_img_exts = {ex for ex, f in img_exts.items() if f in PIL.Image.OPEN}
        img_f_list = []
        for tmp_ext in supported_img_exts:
            img_f_list = img_f_list + list(rgb_dir.glob(f"*{tmp_ext}"))
        img_f_list = sorted(img_f_list)
        n_img_fs = len(img_f_list)

        all_c2w = copy.deepcopy(self.c2w_dict[scene_id])
        all_K = copy.deepcopy(self.K_dict[scene_id])
        assert n_img_fs == all_c2w.shape[0], f"{n_img_fs}, {all_c2w.shape}"
        assert n_img_fs == all_K.shape[0], f"{n_img_fs}, {all_K.shape}"

        src_frame_ids_temporal = []

        # temporally-closest frames
        if tgt_time > 0:
            older_frame_id = int(np.floor(tgt_time))
            src_frame_ids_temporal.append(older_frame_id)

        if tgt_time < n_img_fs - 1:
            # NOTE: we must use this instead of int(np.ceil(tgt_time))
            newer_frame_id = int(np.floor(tgt_time)) + 1
            src_frame_ids_temporal.append(newer_frame_id)

        src_frame_ids_temporal = list(set(src_frame_ids_temporal))

        assert len(set(src_frame_ids_temporal)) == len(
            src_frame_ids_temporal
        ), f"{src_frame_ids_temporal}"
        src_frame_ids_temporal = sorted(src_frame_ids_temporal)

        if len(src_frame_ids_temporal) == 1:
            # We add a placeholder here for convenient processing purposes
            src_frame_ids_temporal.append(src_frame_ids_temporal[0])

        n_actual_temporal = len(src_frame_ids_temporal)

        if n_actual_temporal == 1:
            # We add a placeholder here for convenient processing purposes
            src_frame_ids_temporal.append(src_frame_ids_temporal[0])

        # frames for temporal track
        # older frames
        n_actual_temporal_track_fwd2tgt = 0
        src_frame_ids_temporal_track_fwd2tgt = [
            src_frame_ids_temporal[0]
            for _ in range(self.n_src_views_temporal_track_one_side)
        ]
        if tgt_time > 0:
            tmp_list = np.arange(
                max(0, older_frame_id - self.n_src_views_temporal_track_one_side),
                src_frame_ids_temporal[0],
            )
            src_frame_ids_temporal_track_fwd2tgt[: len(tmp_list)] = tmp_list
            n_actual_temporal_track_fwd2tgt = len(tmp_list)

        # newer frames
        n_actual_temporal_track_bwd2tgt = 0
        src_frame_ids_temporal_track_bwd2tgt = [
            src_frame_ids_temporal[1]
            for _ in range(self.n_src_views_temporal_track_one_side)
        ]
        if tgt_time < n_img_fs - 1:
            tmp_list = np.arange(
                newer_frame_id,
                min(
                    n_img_fs,
                    newer_frame_id + 1 + self.n_src_views_temporal_track_one_side,
                ),
            ).tolist()
            src_frame_ids_temporal_track_bwd2tgt[: len(tmp_list)] = tmp_list
            n_actual_temporal_track_bwd2tgt = len(tmp_list)

        sorted_frame_ids_spatial, sorted_cam_dists_spatial = self.sort_poses_wrt_ref(
            tgt_pose=tgt_c2w,
            ref_poses=np.copy(all_c2w),
            tgt_id=-1,  # no need to mask tgt frame
            dist_method="dist",
        )  # [#frames, ]

        src_frame_ids_spatial = sorted_frame_ids_spatial[: self.n_src_views_spatial]

        assert len(set(src_frame_ids_spatial)) == len(
            src_frame_ids_spatial
        ), f"{src_frame_ids_spatial}"
        src_frame_ids_spatial = sorted(src_frame_ids_spatial)

        frame_ids = np.array(
            [tgt_time, *src_frame_ids_spatial, *src_frame_ids_temporal]
        )
        min_frame_id = np.min(frame_ids)
        max_frame_id = np.max(frame_ids)
        time_range = max_frame_id - min_frame_id

        # dummy
        cur_aug_type = "none"
        cur_square_pad_info = (0, 0)

        dummy_img = np.array(PIL.Image.open(img_f_list[0]))
        tgt_h, tgt_w, _ = dummy_img.shape
        tgt_shape = (tgt_h, tgt_w)

        flat_cam_tgt = np.concatenate(
            ([tgt_h, tgt_w], all_K[0].flatten(), tgt_c2w.flatten())
        ).astype(
            np.float32
        )  # [34, ], 2 + 16 + 16

        (
            rgb_src_spatial,
            flat_cam_src_spatial,
            dyn_rgb_src_spatial,
            static_rgb_src_spatial,
            dyn_mask_src_spatial,
            depth_src_spatial,
            pcl_src_spatial,
        ) = self._extract_information(
            frame_id_list=src_frame_ids_spatial,
            scene_dir=scene_dir,
            img_f_list=img_f_list,
            all_c2w=all_c2w,
            all_K=all_K,
            tgt_shape=tgt_shape,
            use_dyn_mask=True,
            use_depth=True,
        )

        coords_world_homo = np.pad(
            pcl_src_spatial, ((0, 0), (0, 1)), "constant", constant_values=1
        )  # [#pt, 4]
        coords_cam_tgt = np.matmul(
            np.linalg.inv(tgt_c2w), coords_world_homo.T
        ).T  # [#pt, 4]

        depth_range_min = max(1e-16, 0.8 * np.min(coords_cam_tgt[:, 2]))
        depth_range_max = max(2e-16, 1.2 * np.quantile(coords_cam_tgt[:, 2], 0.9))

        depth_range = np.array([depth_range_min, depth_range_max])

        # temporal
        (
            rgb_src_temporal,
            flat_cam_src_temporal,
            dyn_rgb_src_temporal,
            static_rgb_src_temporal,
            dyn_mask_src_temporal,
            depth_src_temporal,
            _,
        ) = self._extract_information(
            frame_id_list=src_frame_ids_temporal,
            scene_dir=scene_dir,
            img_f_list=img_f_list,
            all_c2w=all_c2w,
            all_K=all_K,
            tgt_shape=tgt_shape,
            use_dyn_mask=True,
            use_depth=True,
        )

        # temporal_track_fwd2tgt
        (
            rgb_src_temporal_track_fwd2tgt,
            flat_cam_src_temporal_track_fwd2tgt,
            dyn_rgb_src_temporal_track_fwd2tgt,
            static_rgb_src_temporal_track_fwd2tgt,
            dyn_mask_src_temporal_track_fwd2tgt,
            depth_src_temporal_track_fwd2tgt,
            _,
        ) = self._extract_information(
            frame_id_list=src_frame_ids_temporal_track_fwd2tgt,
            scene_dir=scene_dir,
            img_f_list=img_f_list,
            all_c2w=all_c2w,
            all_K=all_K,
            tgt_shape=tgt_shape,
            use_dyn_mask=True,
            use_depth=True,
        )

        # temporal_track_bwd2tgt
        (
            rgb_src_temporal_track_bwd2tgt,
            flat_cam_src_temporal_track_bwd2tgt,
            dyn_rgb_src_temporal_track_bwd2tgt,
            static_rgb_src_temporal_track_bwd2tgt,
            dyn_mask_src_temporal_track_bwd2tgt,
            depth_src_temporal_track_bwd2tgt,
            _,
        ) = self._extract_information(
            frame_id_list=src_frame_ids_temporal_track_bwd2tgt,
            scene_dir=scene_dir,
            img_f_list=img_f_list,
            all_c2w=all_c2w,
            all_K=all_K,
            tgt_shape=tgt_shape,
            use_dyn_mask=True,
            use_depth=True,
        )

        # read flow
        flow_fwd, flow_fwd_occ_mask = self._read_flow(
            scene_dir=scene_dir,
            img_f_list=img_f_list,
            src_frame_idx=src_frame_ids_temporal[0],
            tgt_frame_idx=src_frame_ids_temporal[1],
            tgt_shape=tgt_shape,
            occ_thres=self.flow_consist_thres,
        )

        flow_bwd, flow_bwd_occ_mask = self._read_flow(
            scene_dir=scene_dir,
            img_f_list=img_f_list,
            src_frame_idx=src_frame_ids_temporal[1],
            tgt_frame_idx=src_frame_ids_temporal[0],
            tgt_shape=tgt_shape,
            occ_thres=self.flow_consist_thres,
        )

        # fmt: off
        ret_dict = {
            "scene_id": scene_id,
            "seq_ids": torch.LongTensor(frame_ids),  # [N]
            # rgb src spatial
            "rgb_src_spatial": torch.FloatTensor(rgb_src_spatial),  # [N, H, W, 3]
            "dyn_rgb_src_spatial": torch.FloatTensor(dyn_rgb_src_spatial),  # [N, H, W, 3]
            "static_rgb_src_spatial": torch.FloatTensor(static_rgb_src_spatial),  # [N, H, W, 3]
            # rgb src temporal
            "n_actual_temporal": torch.LongTensor([n_actual_temporal]),  # [1,]
            "rgb_src_temporal": torch.FloatTensor(rgb_src_temporal),  # [2, H, W, 3]
            "dyn_rgb_src_temporal": torch.FloatTensor(dyn_rgb_src_temporal),  # [2, H, W, 3]
            "static_rgb_src_temporal": torch.FloatTensor(static_rgb_src_temporal),  # [2, H, W, 3]
            # rgb src temporal_track_fwd2tgt
            "n_actual_temporal_track_fwd2tgt": torch.LongTensor([n_actual_temporal_track_fwd2tgt]),  # [1,]
            "rgb_src_temporal_track_fwd2tgt": torch.FloatTensor(rgb_src_temporal_track_fwd2tgt),  # [N, H, W, 3]
            "dyn_rgb_src_temporal_track_fwd2tgt": torch.FloatTensor(dyn_rgb_src_temporal_track_fwd2tgt),  # [N, H, W, 3]
            "static_rgb_src_temporal_track_fwd2tgt": torch.FloatTensor(static_rgb_src_temporal_track_fwd2tgt),  # [N, H, W, 3]
            # rgb src temporal_track_bwd2tgt
            "n_actual_temporal_track_bwd2tgt": torch.LongTensor([n_actual_temporal_track_bwd2tgt]),  # [1,]
            "rgb_src_temporal_track_bwd2tgt": torch.FloatTensor(rgb_src_temporal_track_bwd2tgt),  # [N, H, W, 3]
            "dyn_rgb_src_temporal_track_bwd2tgt": torch.FloatTensor(dyn_rgb_src_temporal_track_bwd2tgt),  # [N, H, W, 3]
            "static_rgb_src_temporal_track_bwd2tgt": torch.FloatTensor(static_rgb_src_temporal_track_bwd2tgt),  # [N, H, W, 3]
            # mask
            "dyn_mask_src_spatial": torch.FloatTensor(dyn_mask_src_spatial)[..., None],  # [N, H, W, 1]
            "dyn_mask_src_temporal": torch.FloatTensor(dyn_mask_src_temporal)[..., None],  # [2, H, W, 1]
            "dyn_mask_src_temporal_track_fwd2tgt": torch.FloatTensor(dyn_mask_src_temporal_track_fwd2tgt)[..., None],  # [N, H, W, 1]
            "dyn_mask_src_temporal_track_bwd2tgt": torch.FloatTensor(dyn_mask_src_temporal_track_bwd2tgt)[..., None],  # [N, H, W, 1]
            # flow
            "flow_fwd": torch.FloatTensor(flow_fwd),  # [H, W, 2]
            "flow_fwd_occ_mask": torch.FloatTensor(flow_fwd_occ_mask)[..., None],  # [H, W, 1]
            "flow_bwd": torch.FloatTensor(flow_bwd),  # [H, W, 2]
            "flow_bwd_occ_mask": torch.FloatTensor(flow_bwd_occ_mask)[..., None],  # [H, W, 1]
            # cam info
            "flat_cam_tgt": torch.FloatTensor(flat_cam_tgt),  # [34,]
            "flat_cam_src_spatial": torch.FloatTensor(flat_cam_src_spatial),  # [N, 34]
            "flat_cam_src_temporal": torch.FloatTensor(flat_cam_src_temporal),  # [2, 34]
            "flat_cam_src_temporal_track_fwd2tgt": torch.FloatTensor(flat_cam_src_temporal_track_fwd2tgt),  # [N, 34]
            "flat_cam_src_temporal_track_bwd2tgt": torch.FloatTensor(flat_cam_src_temporal_track_bwd2tgt),  # [N, 34]
            # depth
            "depth_src_temporal": torch.FloatTensor(depth_src_temporal)[..., None],  # [2, H, W, 1]
            "depth_src_temporal_track_fwd2tgt": torch.FloatTensor(depth_src_temporal_track_fwd2tgt)[..., None],  # [N, H, W, 1]
            "depth_src_temporal_track_bwd2tgt": torch.FloatTensor(depth_src_temporal_track_bwd2tgt)[..., None],  # [N, H, W, 1]
            "depth_range": torch.FloatTensor(depth_range),
            # time
            "time_tgt": torch.FloatTensor([tgt_time]),  # [1, ]
            "time_src_temporal": torch.FloatTensor(src_frame_ids_temporal),  # [2, ]
            "time_src_temporal_track_fwd2tgt": torch.FloatTensor(src_frame_ids_temporal_track_fwd2tgt),  # [N, ]
            "time_src_temporal_track_bwd2tgt": torch.FloatTensor(src_frame_ids_temporal_track_bwd2tgt),  # [N, ]
            # misc
            "misc": {
                "scene_id": scene_id,
                "tgt_time": tgt_time,
                "tgt_idx": tgt_idx,
            },
        }
        # fmt: on

        return ret_dict

    def _extract_information(
        self,
        *,
        scene_dir,
        img_f_list,
        frame_id_list,
        all_c2w,
        all_K,
        tgt_shape,
        use_dyn_mask=False,
        use_depth=False,
    ):
        rgb_src = []
        flat_cam_src = []
        dyn_rgb_src = []
        static_rgb_src = []
        dyn_mask_src = []
        depth_src = []
        pcl_src = []

        pcl_dummpy = np.zeros((1, 3))

        for tmp_src_id in frame_id_list:
            tmp_img_f = img_f_list[tmp_src_id]
            (
                tmp_rgb_src,
                tmp_flat_cam_src,
                tmp_extra_info,
            ) = self._process_for_single_src_view(
                scene_dir=scene_dir,
                img_f=tmp_img_f,
                tgt_shape=tgt_shape,
                c2w=all_c2w[tmp_src_id, ...],
                K=all_K[tmp_src_id, :],
                use_dyn_mask=use_dyn_mask,
                use_depth=use_depth,
            )

            rgb_src.append(tmp_rgb_src)
            flat_cam_src.append(tmp_flat_cam_src)
            dyn_rgb_src.append(tmp_extra_info["dyn_rgb"])
            static_rgb_src.append(tmp_extra_info["static_rgb"])
            dyn_mask_src.append(tmp_extra_info["dyn_mask"])
            depth_src.append(tmp_extra_info["depth"])
            if use_depth:
                pcl_src.append(tmp_extra_info["pcl"])
            else:
                pcl_src.append(pcl_dummpy)

        rgb_src = np.stack(rgb_src, axis=0)
        flat_cam_src = np.stack(flat_cam_src, axis=0)
        dyn_rgb_src = np.stack(dyn_rgb_src, axis=0)
        static_rgb_src = np.stack(static_rgb_src, axis=0)
        dyn_mask_src = np.stack(dyn_mask_src, axis=0)
        depth_src = np.stack(depth_src, axis=0)
        pcl_src = np.concatenate(pcl_src, axis=0)

        return (
            rgb_src,
            flat_cam_src,
            dyn_rgb_src,
            static_rgb_src,
            dyn_mask_src,
            depth_src,
            pcl_src,
        )

    def _process_for_single_src_view(
        self,
        *,
        scene_dir,
        img_f,
        tgt_shape,
        c2w,
        K,
        use_dyn_mask=False,
        use_depth=False,
    ):
        rgb = np.array(PIL.Image.open(img_f))

        tgt_h, tgt_w = tgt_shape
        if rgb.shape[0] != tgt_h or rgb.shape[1] != tgt_w:
            rgb = cv2.resize(
                rgb,
                dsize=(tgt_w, tgt_h),
                interpolation=cv2.INTER_AREA,
            )

        extra_info = {}

        if use_dyn_mask or use_depth:
            if use_dyn_mask:
                dyn_mask = self._read_mask(scene_dir, img_f.stem, tgt_h, tgt_w)

                dyn_mask = dyn_mask.astype(np.float32)  # [H, W]

                extra_info["dyn_mask"] = dyn_mask

            if use_depth:
                depth = self._read_depth(scene_dir, img_f.stem)  # [H, W]

                if depth.shape[0] != tgt_h or depth.shape[1] != tgt_w:
                    depth = cv2.resize(
                        depth, (tgt_w, tgt_h), interpolation=cv2.INTER_NEAREST
                    )

                extra_info["depth"] = depth

        rgb = modify_rgb_range(rgb, src_range="0_255", tgt_range=self.rgb_range)

        if use_dyn_mask:
            # NOTE: we compute dyn_rgb here after rgb is converted to target range
            extra_info["dyn_rgb"] = rgb * dyn_mask[..., None].astype(np.float32)
            extra_info["static_rgb"] = rgb * (
                1 - dyn_mask[..., None].astype(np.float32)
            )

        cur_h, cur_w, _ = rgb.shape
        flat_cam_info = np.concatenate(
            ([cur_h, cur_w], K.flatten(), c2w.flatten())
        ).astype(
            np.float32
        )  # [34, ], 2 + 16 + 16

        if use_depth:
            pcl = self._compute_pcl(cur_h, cur_w, K, c2w, extra_info["depth"])
            extra_info["pcl"] = pcl

        return rgb, flat_cam_info, extra_info

    def _compute_pcl(self, h, w, K, c2w, depth):
        K_torch = torch.FloatTensor(K)[None, ...]
        c2w_torch = torch.FloatTensor(c2w)[None, ...]
        rays_o, rays_d, _, _ = self._get_rays_single_image(
            h, w, K_torch, c2w_torch
        )  # [H x W, 3]
        pcl = rays_o + rays_d * depth.reshape((-1, 1))
        return pcl

    def _read_mask(self, scene_dir, fname, tgt_h, tgt_w):
        dyn_mask_f = scene_dir / f"masks/final/{fname}_final.png"

        dyn_mask = np.array(
            PIL.Image.open(dyn_mask_f)
        )  # bool. True indicates dynamic part

        if dyn_mask.shape[0] != tgt_h or dyn_mask.shape[1] != tgt_w:
            dyn_mask = np.array(
                PIL.Image.fromarray(dyn_mask).resize(
                    (tgt_w, tgt_h), resample=PIL.Image.Resampling.NEAREST
                )
            )

        return dyn_mask

    def _read_depth(self, scene_dir, fname):
        depth_f = scene_dir / f"depths/{fname}.npz"
        depth = np.load(depth_f)["depth"]  # [H, W]
        return depth

    def _read_flow(
        self,
        *,
        scene_dir,
        img_f_list,
        src_frame_idx,
        tgt_frame_idx,
        tgt_shape,
        occ_thres=1,
    ):
        if src_frame_idx == tgt_frame_idx:
            flow = np.zeros(list(tgt_shape) + [2], dtype=np.float32)
            mask_occ = np.zeros(tgt_shape, dtype=np.float32)
        else:
            flow_interval = abs(tgt_frame_idx - src_frame_idx)
            src_fname = img_f_list[src_frame_idx].stem
            tgt_fname = img_f_list[tgt_frame_idx].stem
            flow_f = (
                scene_dir
                / f"flows/interval_{flow_interval}"
                / f"{src_fname}_{tgt_fname}.npz"
            )
            flow_info = np.load(flow_f)

            flow = flow_info["flow"]  # [H, W, 2]
            coord_diff = flow_info["coord_diff"]  # [H, W, 2]
            mask_occ = (np.sum(np.abs(coord_diff), axis=2) > occ_thres).astype(
                np.float32
            )  # [H, W], bool

            assert flow.shape[0] == tgt_shape[0], f"{flow.shape}, {tgt_shape}"
            assert flow.shape[1] == tgt_shape[1], f"{flow.shape}, {tgt_shape}"

        return flow, mask_occ
