import copy
import tqdm
import trimesh
import logging
import pathlib
import trimesh
import numpy as np

import torch

from pgdvs.utils.vis_utils import draw_cam_mesh
from pgdvs.datasets.nvidia_eval import NvidiaDynEvaluationDataset
from pgdvs.utils.geometry import linear_pose_interp, recenter_poses


DEBUG_DIR = pathlib.Path(__file__).absolute().parent.parent.parent / "debug"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)


ALL_SCENE_IDS_NVIDIA_DYN = [
    "Balloon1",
    "Balloon2",
    "Jumping",
    "Playground",
    "Skating",
    "Truck",
    "Umbrella",
    "dynamicFace",
]

N_CAMS = 12

TGT_HEIGHT = 288

FLAG_RESCALE_POSES = False
FLAG_CENTER_POSES = False

FLAG_SAVE_CAM_MESH = False

N_BT_REPS = 8


LOGGER = logging.getLogger(__name__)


class NvidiaDynVisualizationDataset(NvidiaDynEvaluationDataset):
    dataset_name = "NVIDIA_Dyn Visualization"
    dataset_fname = "nvidia_vis"

    def __init__(
        self,
        *,
        data_root,
        raw_data_dir,
        depth_data_dir,
        mask_data_dir,
        flow_data_dir,
        max_hw,
        mode,
        rgb_range="0_1",
        use_aug=False,
        scene_ids=None,
        n_src_views_spatial=10,
        n_src_views_temporal_track_one_side=5,
        use_zoe_depth="none",
        zoe_depth_data_f=None,
        flow_consist_thres=1.0,
        # vis
        vis_center_time=50,
        n_render_frames=200,
        vis_time_interval=10,
        vis_bt_max_disp=32,
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

        self.raw_data_dir = pathlib.Path(data_root) / raw_data_dir
        assert self.raw_data_dir.exists(), self.raw_data_dir

        self.depth_data_dir = pathlib.Path(data_root) / depth_data_dir
        assert self.depth_data_dir.exists(), self.depth_data_dir

        self.mask_data_dir = pathlib.Path(data_root) / mask_data_dir
        assert self.mask_data_dir.exists(), self.mask_data_dir

        self.flow_data_dir = pathlib.Path(data_root) / flow_data_dir
        assert self.flow_data_dir.exists(), self.flow_data_dir

        self.flow_consist_thres = flow_consist_thres

        zoe_k_dict = {}
        for zoe_type in ["n", "k", "nk"]:
            for zoe_k in [
                "me_med_share",
                "me_med_indiv",
                "me_trim_share",
                "me_trim_indiv",
            ]:
                zoe_k_dict[f"{zoe_type}_{zoe_k}"] = (zoe_type, zoe_k)
        self.zoe_k_dict = zoe_k_dict

        assert use_zoe_depth in ["none", "moe"] + list(
            zoe_k_dict.keys()
        ), f"{use_zoe_depth}, {zoe_k_dict.keys()}"

        assert use_zoe_depth in ["none", "moe"] + list(
            zoe_k_dict.keys()
        ), f"{use_zoe_depth}, {zoe_k_dict.keys()}"

        self.zoe_depth_data_f = None
        self.use_zoe_depth = use_zoe_depth
        if self.use_zoe_depth != "none":
            self.zoe_depth_data_f = pathlib.Path(data_root) / zoe_depth_data_f
            assert self.zoe_depth_data_f.exists(), self.zoe_depth_data_f

        self.aug_types = [
            "none",
            # "rot90",
            # "rot180",
            # "rot270",
            # "flip_horizontal",
            # "flip_vertical",
        ]

        LOGGER.info(
            f"[{self.dataset_name}] [{self.mode}] raw_data_dir: {self.raw_data_dir}; \n"
            f"depth_data_dir: {self.depth_data_dir}; mask_data_dir: {self.mask_data_dir}.\n"
        )

        assert self.mode in ["vis"], f"{self.mode}"

        LOGGER.info(f"[{self.dataset_name}] [{self.mode}] use_aug: {use_aug}\n")

        LOGGER.info(f"[{self.dataset_name}] [{self.mode}]\n")

        if scene_ids is None:
            scene_ids = ALL_SCENE_IDS_NVIDIA_DYN

        LOGGER.info(
            f"[{self.dataset_name}] [{self.mode}] scene_ids: {scene_ids}, {type(scene_ids)}\n"
        )

        c2w_dict = {}
        hwf_dict = {}
        all_vis_poses = []

        for tmp_scene_id in scene_ids:
            tmp_scene_dir = self.raw_data_dir / tmp_scene_id / "dense"

            # NOTE: here we use poses_bounds_cvd because we use the DynIBaR-processed depths,
            # which is bonded to the poses_bounds_cvd.npy.
            cam_f = tmp_scene_dir / "poses_bounds_cvd.npy"
            poses_arr = np.load(cam_f, allow_pickle=True)  # [#frames, 17]

            n_img_fs = poses_arr.shape[0]

            render_time_list = np.linspace(
                max(0, vis_center_time - vis_time_interval),
                min(n_img_fs - 2, vis_center_time + vis_time_interval),
                n_render_frames,
            ).tolist()

            poses = (
                poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
            )  # [3, 5, #frame]
            bds = poses_arr[:, -2:].astype(np.float32)  # [#frame, 2]

            # Correct rotation matrix ordering and move variable dim to axis 0
            # https://github.com/Fyusion/LLFF/blob/c6e27b1ee59cb18f054ccb0f87a90214dbe70482/README.md#using-your-own-poses-without-running-colmap
            poses = np.concatenate(
                [poses[:, 1:2, :], -poses[:, 0:1, :], poses[:, 2:, :]], 1
            )  # [down, right, back] -> [+X right, +Y up, +Z back]
            poses = np.moveaxis(poses, -1, 0).astype(
                np.float32
            )  # [3, 5, #frame] -> [#frame, 3, 5]

            all_hwf = poses[:, :, 4]  # [#frame, 3]
            homo_placeholder = np.zeros((n_img_fs, 1, 4))
            homo_placeholder[..., 3] = 1
            all_c2w = np.concatenate(
                (poses[:, :, :4], homo_placeholder), axis=1
            )  # [#frames, 3, 4] -> [#frames, 4, 4]

            # [right, up, back] (LLFF) -> [right, down, forward] (OpenCV's convention)
            all_c2w[..., 1:3] *= -1

            if FLAG_SAVE_CAM_MESH:
                draw_set_poses(
                    all_c2w[:24, ...],
                    DEBUG_DIR / "vis_poses_raw.ply",
                    tmp_coord=1.0,
                )

            bd_factor = 0.9
            if FLAG_RESCALE_POSES:
                # TODO: do we need this though: https://github.com/zhengqili/Neural-Scene-Flow-Fields/blob/d4001759a39b056c95d8bc22da34b10b4fb85afb/nsff_exp/load_llff.py#L263-L271
                # Rescale if bd_factor is provided
                # - https://github.com/bmild/nerf/issues/34
                # - Appendix C in https://arxiv.org/pdf/2003.08934.pdf
                pose_sc = 1.0 / (np.percentile(bds[:, 0], 5) * bd_factor)
                bt_disp_sc = 1.0
            else:
                pose_sc = 1.0
                bt_disp_sc = 1.0 / (np.percentile(bds[:, 0], 5) * bd_factor)

            all_c2w[:, :3, 3] *= pose_sc  # camera position

            if FLAG_SAVE_CAM_MESH:
                draw_set_poses(
                    all_c2w[:24, ...],
                    DEBUG_DIR / "vis_poses_sc.ply",
                    tmp_coord=1.0,
                )

            if FLAG_CENTER_POSES:
                all_c2w = recenter_poses(all_c2w)  # [#frame, 3, 5]

                if FLAG_SAVE_CAM_MESH:
                    draw_set_poses(
                        all_c2w[:24, ...],
                        DEBUG_DIR / "vis_poses_centered.ply",
                        tmp_coord=1.0,
                    )

            raw_resize_dir_list = list(tmp_scene_dir.glob(f"images_*x{TGT_HEIGHT}"))
            assert (
                len(raw_resize_dir_list) == 1
            ), f"{tmp_scene_dir}, {raw_resize_dir_list}"
            raw_w, raw_h = raw_resize_dir_list[0].stem.split("_")[1].split("x")
            raw_w = int(raw_w)
            raw_h = int(raw_h)
            all_hwf[:, 0] = raw_h
            all_hwf[:, 1] = raw_w

            bt_poses = create_bt_poses(
                all_hwf[0, 2],
                num_frames=len(render_time_list) // N_BT_REPS,
                sc=bt_disp_sc,
                max_disp=vis_bt_max_disp,
            )
            bt_poses = bt_poses * (N_BT_REPS + 1)  # * int(10 * n_frames / 24)

            c2w_dict[tmp_scene_id] = np.copy(all_c2w)
            hwf_dict[tmp_scene_id] = np.copy(all_hwf)

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

        self.zip_obj = None  # enable lazy read
        self.zoe_depth_zip_obj = None

        self.c2w_dict = c2w_dict
        self.hwf_dict = hwf_dict

        self.valid_fs = all_vis_poses

    def _get_item_func(self, index):
        scene_id, scene_dir, tgt_time, tgt_idx, tgt_c2w, c2w_sc = self.valid_fs[index]

        all_c2w = copy.deepcopy(self.c2w_dict[scene_id])
        all_hwf = copy.deepcopy(self.hwf_dict[scene_id])
        n_img_fs = all_c2w.shape[0]
        assert n_img_fs == all_hwf.shape[0], f"{n_img_fs}, {all_hwf.shape}"

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

        assert (
            self.n_src_views_spatial < N_CAMS * 2
        ), f"{self.n_src_views_spatial}, {N_CAMS * 2}"

        src_frame_ids_spatial_pool = np.arange(
            max(0, src_frame_ids_temporal[0] - N_CAMS),
            min(n_img_fs, src_frame_ids_temporal[1] + N_CAMS),
        ).tolist()

        sorted_frame_ids_spatial, sorted_cam_dists_spatial = self.sort_poses_wrt_ref(
            tgt_pose=tgt_c2w,
            ref_poses=np.copy(
                all_c2w[src_frame_ids_spatial_pool, ...]
            ),  # we only need the first 12 cameras
            tgt_id=-1,  # no need to mask tgt frame
            dist_method="dist",
        )  # [#frames, ]

        src_frame_ids_spatial = [
            src_frame_ids_spatial_pool[_] for _ in sorted_frame_ids_spatial
        ]

        src_frame_ids_spatial = src_frame_ids_spatial[: self.n_src_views_spatial]

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

        # process RGB
        mono_dir_list = list(pathlib.Path(scene_dir).glob(f"images_*x{TGT_HEIGHT}"))
        assert len(mono_dir_list) == 1, f"{mono_dir_list}"
        mono_dirname = mono_dir_list[0].name
        raw_w, raw_h = mono_dirname.split("images_")[1].split("x")
        raw_h = int(raw_h)
        raw_w = int(raw_w)

        assert raw_h == TGT_HEIGHT, f"{raw_h}, {TGT_HEIGHT}"

        tgt_shape = (raw_h, raw_w)

        # dummy
        cur_aug_type = "none"
        cur_square_pad_info = (0, 0)

        # NOTE: since we do not have any augmentation, (cur_h, cur_w) are just tgt_shape
        aug_shape = tgt_shape

        assert cur_aug_type == "none", f"{cur_aug_type}"

        aug_c2w, aug_K = self._compute_cam_info(
            aug_type=cur_aug_type, c2w=tgt_c2w, hwf=all_hwf[0], tgt_shape=tgt_shape
        )

        flat_cam_tgt = np.concatenate(
            ([aug_shape[0], aug_shape[1]], aug_K.flatten(), aug_c2w.flatten())
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
            scene_id=scene_id,
            all_c2w=all_c2w,
            all_hwf=all_hwf,
            tgt_shape=tgt_shape,
            aug_type=cur_aug_type,
            square_pad_info=cur_square_pad_info,
            use_dyn_mask=True,
            use_depth=True,
        )

        coords_world_homo = np.pad(
            pcl_src_spatial, ((0, 0), (0, 1)), "constant", constant_values=1
        )  # [#pt, 4]
        coords_cam_tgt = np.matmul(
            np.linalg.inv(aug_c2w), coords_world_homo.T
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
            scene_id=scene_id,
            all_c2w=all_c2w,
            all_hwf=all_hwf,
            tgt_shape=tgt_shape,
            aug_type=cur_aug_type,
            square_pad_info=cur_square_pad_info,
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
            scene_id=scene_id,
            all_c2w=all_c2w,
            all_hwf=all_hwf,
            tgt_shape=tgt_shape,
            aug_type=cur_aug_type,
            square_pad_info=cur_square_pad_info,
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
            scene_id=scene_id,
            all_c2w=all_c2w,
            all_hwf=all_hwf,
            tgt_shape=tgt_shape,
            aug_type=cur_aug_type,
            square_pad_info=cur_square_pad_info,
            use_dyn_mask=True,
            use_depth=True,
        )

        # read flow
        flow_fwd, flow_fwd_occ_mask = self._read_flow(
            scene_id=scene_id,
            src_frame_id=src_frame_ids_temporal[0],
            tgt_frame_id=src_frame_ids_temporal[1],
            tgt_shape=tgt_shape,
            occ_thres=self.flow_consist_thres,
            aug_type=cur_aug_type,
            square_pad_info=cur_square_pad_info,
            crop_info=None,
        )

        flow_bwd, flow_bwd_occ_mask = self._read_flow(
            scene_id=scene_id,
            src_frame_id=src_frame_ids_temporal[1],
            tgt_frame_id=src_frame_ids_temporal[0],
            tgt_shape=tgt_shape,
            occ_thres=self.flow_consist_thres,
            aug_type=cur_aug_type,
            square_pad_info=cur_square_pad_info,
            crop_info=None,
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

    def _get_img_f_for_src_view(self, scene_id, src_frame_id):
        # NOTE: for DynIBaR, the monocular video is selected from the original multiview vidoes in a round manner.
        # Namely, for the i-th frame, DynIBaR chooses the (i % 12)-th camera.
        # Therefore, for the monocular video, the frame and the camera has the same index.
        # Ref: https://github.com/google/dynibar/blob/1b15d7fb926cb997803514a344ce0a4c9086d49c/eval_nvidia.py#L317
        tmp_cam_id = src_frame_id % N_CAMS
        tmp_scene_dir = self.raw_data_dir / scene_id / "dense"
        tmp_img_f = (
            tmp_scene_dir
            / f"mv_images/{src_frame_id:05d}"
            / f"cam{(tmp_cam_id + 1):02d}.jpg"
        )  # index starts from 1
        return tmp_img_f


def draw_set_poses(all_c2w, save_f, tmp_coord=1.0):
    n = all_c2w.shape[0]

    all_verts = []
    all_colors = []
    for i in range(n):
        tmp_c2w = all_c2w[i, :4, :4]  # [3, 4]
        tmp_verts, tmp_colors = draw_cam_mesh(
            np.linalg.inv(tmp_c2w), None, tmp_coord=tmp_coord, flag_save=False
        )
        tmp_colors = (tmp_colors.astype(float) * (i + 1) / (n + 2)).astype(np.uint8)
        all_verts.append(tmp_verts)
        all_colors.append(tmp_colors)

    all_verts = np.concatenate(all_verts, axis=0)
    all_colors = np.concatenate(all_colors, axis=0)

    frame = trimesh.points.PointCloud(vertices=all_verts, colors=all_colors)
    _ = frame.export(save_f)


def create_bt_poses(focal, num_frames=40, sc=None, max_disp=32):
    # Modified from https://github.com/zhengqili/Neural-Scene-Flow-Fields/blob/d4001759a39b056c95d8bc22da34b10b4fb85afb/nsff_exp/load_llff.py#L394

    if sc is not None:
        max_disp = max_disp / sc

    max_trans = (
        max_disp / focal
    )  # Maximum camera translation to satisfy max_disp parameter
    output_poses = []

    for i in range(num_frames):
        x_trans = max_trans * np.sin(2.0 * np.pi * float(i) / float(num_frames))
        y_trans = max_trans * np.cos(2.0 * np.pi * float(i) / float(num_frames)) / 2.0
        z_trans = 0.0

        i_pose = np.concatenate(
            [
                np.concatenate(
                    [np.eye(3), np.array([x_trans, y_trans, z_trans])[:, np.newaxis]],
                    axis=1,
                ),
                np.array([0.0, 0.0, 0.0, 1.0])[np.newaxis, :],
            ],
            axis=0,
        )

        i_pose = np.linalg.inv(i_pose)
        output_poses.append(i_pose)

    return output_poses
