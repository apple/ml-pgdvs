import cv2
import logging
import pathlib
import PIL.Image
import numpy as np
from collections import defaultdict
from sklearn.cluster import KMeans

import torch

from pgdvs.utils.rendering import modify_rgb_range
from pgdvs.utils.vis_utils import draw_cam_mesh
from pgdvs.datasets.base import BaseDataset
from pgdvs.datasets.dycheck_utils import iPhoneParser


DEBUG_DIR = pathlib.Path(__file__).absolute().parent.parent.parent / "debug"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)


ALL_SCENE_IDS_DYCHECK_IPHONE = [
    "apple",
    "block",
    "paper-windmill",
    "space-out",
    "spin",
    "teddy",
    "wheel",
]


LOGGER = logging.getLogger(__name__)


class DyCheckiPhoneEvaluationDataset(BaseDataset):
    dataset_name = "DyCheck iPhone Eval"
    dataset_fname = "dycheck_iphone_eval"

    def __init__(
        self,
        *,
        data_root,
        raw_data_dir,
        mask_data_dir,
        flow_data_dir,
        max_hw,
        mode,
        rgb_range="0_1",
        use_aug=False,
        scene_ids=None,
        spatial_src_view_type="clustered",
        n_src_views_spatial=10,
        n_src_views_spatial_cluster=None,
        n_src_views_temporal_track_one_side=5,
        flow_consist_thres=1.0,
    ):
        self.mode = mode

        assert (
            max_hw == -1
        ), f"We enforce to use raw resolution. However, we receive max_hw of {max_hw}"
        self.max_hw = max_hw

        assert not use_aug
        self.use_aug = use_aug

        self.rgb_range = rgb_range
        self.n_src_views_spatial = n_src_views_spatial
        self.spatial_src_view_type = spatial_src_view_type
        if n_src_views_spatial_cluster is None:
            n_src_views_spatial_cluster = self.n_src_views_spatial
        self.n_src_views_spatial_cluster = n_src_views_spatial_cluster
        self.n_src_views_temporal_track_one_side = n_src_views_temporal_track_one_side

        LOGGER.info(
            f"[{self.dataset_name}] [{self.mode}] spatial_src_view_type: {self.spatial_src_view_type}. \n"
        )

        assert self.max_hw == -1, f"{self.max_hw}"

        self.use_normalized_K = False

        self.raw_data_dir = pathlib.Path(data_root) / raw_data_dir
        assert self.raw_data_dir.exists(), self.raw_data_dir

        self.mask_data_dir = pathlib.Path(data_root) / mask_data_dir
        assert self.mask_data_dir.exists(), self.mask_data_dir

        self.flow_data_dir = pathlib.Path(data_root) / flow_data_dir
        assert self.flow_data_dir.exists(), self.flow_data_dir

        self.flow_consist_thres = flow_consist_thres

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
            f"mask_data_dir: {self.mask_data_dir}.\n"
        )

        assert self.mode in ["eval"], f"{self.mode}"

        LOGGER.info(f"[{self.dataset_name}] [{self.mode}] use_aug: {use_aug}\n")

        LOGGER.info(f"[{self.dataset_name}] [{self.mode}]\n")

        if scene_ids is None:
            scene_ids = ALL_SCENE_IDS_DYCHECK_IPHONE

        LOGGER.info(
            f"[{self.dataset_name}] [{self.mode}] scene_ids: {scene_ids}, {type(scene_ids)}\n"
        )

        self.parser_dict = {}
        self.train_info_dict = {}
        all_data = []
        for scene_id in scene_ids:
            self.parser_dict[scene_id] = iPhoneParser(
                scene_id, data_root=self.raw_data_dir
            )

            train_frame_names, train_time_ids, train_camera_ids = self.parser_dict[
                scene_id
            ].load_split("train")
            n_train_frame_names = len(train_frame_names)
            n_train_time_ids = len(train_time_ids)
            n_train_camera_ids = len(train_camera_ids)
            assert (
                n_train_frame_names == n_train_time_ids
            ), f"{n_train_frame_names}, {n_train_time_ids}"
            assert (
                n_train_frame_names == n_train_camera_ids
            ), f"{n_train_frame_names}, {n_train_camera_ids}"

            # make sure the training time_ids are consecutive
            min_time_id = min(train_time_ids)
            max_time_id = max(train_time_ids)
            assert n_train_time_ids == (
                max_time_id - min_time_id + 1
            ), f"{n_train_time_ids}, {min_time_id}, {max_time_id}"

            self.train_info_dict[scene_id] = {
                "frame_names": train_frame_names,
                "time_ids": train_time_ids.astype(int),
                "camera_ids": train_camera_ids.astype(int),
                "unique_ids": [],
            }

            all_train_c2w = []
            for i in range(n_train_frame_names):
                self.train_info_dict[scene_id]["unique_ids"].append(
                    (train_frame_names[i], train_time_ids[i], train_camera_ids[i])
                )

                tmp_w2c = (
                    self.parser_dict[scene_id]
                    .load_camera(train_time_ids[i], train_camera_ids[i])
                    .extrin
                )  # [4, 4]
                all_train_c2w.append(np.linalg.inv(tmp_w2c))

            self.train_info_dict[scene_id]["train_c2w"] = np.array(
                all_train_c2w
            )  # [#frame, 4, 4]

            val_frame_names, val_time_ids, val_camera_ids = self.parser_dict[
                scene_id
            ].load_split("val")
            n_val_frame_names = len(val_frame_names)
            n_val_time_ids = len(val_time_ids)
            n_val_camera_ids = len(val_camera_ids)
            assert (
                n_val_frame_names == n_val_time_ids
            ), f"{n_val_frame_names}, {n_val_time_ids}"
            assert (
                n_val_frame_names == n_val_camera_ids
            ), f"{n_val_frame_names}, {n_val_camera_ids}"

            for i in range(n_val_frame_names):
                all_data.append(
                    (scene_id, val_frame_names[i], val_time_ids[i], val_camera_ids[i])
                )

        assert len(set(all_data)) == len(
            all_data
        ), f"{len(set(all_data))}, {len(all_data)}"

        # NOTE: must sort here to ensure same order across workers
        self.valid_fs = sorted(list(set(all_data)))

        LOGGER.info(
            f"[{self.dataset_name}] [{self.mode}] #final_valid_fs: {len(self.valid_fs)}\n"
        )

        self.zip_obj = None  # enable lazy read
        self.zoe_depth_zip_obj = None

    def get_train_cam_id(self, scene_id):
        train_cam_ids = self.train_info_dict[scene_id]["camera_ids"]
        assert len(set(train_cam_ids)) == 1, f"{set(train_cam_ids)}"
        return train_cam_ids[0]

    def __len__(self):
        return len(self.valid_fs)

    def __getitem__(self, index):
        return self._get_item_func(index)

    def _get_item_func(self, index):
        scene_id, tgt_frame_name, tgt_time_id, tgt_cam_id = self.valid_fs[index]

        assert (tgt_frame_name, tgt_time_id, tgt_cam_id) not in self.train_info_dict[
            scene_id
        ]["unique_ids"], f"{scene_id}, {(tgt_frame_name, tgt_time_id, tgt_cam_id)}"

        tgt_rgb, _, tgt_K, tgt_w2c = self._extract_view_info(
            self.parser_dict[scene_id], tgt_time_id, tgt_cam_id, view_in_train=False
        )

        min_train_time_id = min(self.train_info_dict[scene_id]["time_ids"])
        max_train_time_id = max(self.train_info_dict[scene_id]["time_ids"])

        src_frame_ids_temporal = []

        # temporally-closest frames
        if tgt_time_id in self.train_info_dict[scene_id]["time_ids"]:
            src_frame_ids_temporal.append(tgt_time_id)
        else:
            if tgt_time_id > min_train_time_id:
                older_frame_id = max(
                    [
                        _
                        for _ in self.train_info_dict[scene_id]["time_ids"]
                        if _ < tgt_time_id
                    ]
                )
                src_frame_ids_temporal.append(older_frame_id)

            if tgt_time_id < max_train_time_id:
                newer_frame_id = min(
                    [
                        _
                        for _ in self.train_info_dict[scene_id]["time_ids"]
                        if _ > tgt_time_id
                    ]
                )
                src_frame_ids_temporal.append(newer_frame_id)

        assert len(set(src_frame_ids_temporal)) == len(
            src_frame_ids_temporal
        ), f"{src_frame_ids_temporal}"
        src_frame_ids_temporal = sorted(src_frame_ids_temporal)

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

        if tgt_time_id > min_train_time_id:
            tmp_list = np.arange(
                max(
                    min_train_time_id,
                    src_frame_ids_temporal[0]
                    - self.n_src_views_temporal_track_one_side,
                ),
                src_frame_ids_temporal[0],
            ).tolist()
            n_actual_temporal_track_fwd2tgt = len(tmp_list)
            for i in range(n_actual_temporal_track_fwd2tgt):
                if tmp_list[i] in self.train_info_dict[scene_id]["time_ids"]:
                    src_frame_ids_temporal_track_fwd2tgt[
                        -(n_actual_temporal_track_fwd2tgt - i)
                    ] = tmp_list[i]

        # newer frames
        n_actual_temporal_track_bwd2tgt = 0
        src_frame_ids_temporal_track_bwd2tgt = [
            src_frame_ids_temporal[1]
            for _ in range(self.n_src_views_temporal_track_one_side)
        ]

        if tgt_time_id < max_train_time_id:
            tmp_list = np.arange(
                src_frame_ids_temporal[1] + 1,
                min(
                    max_train_time_id + 1,
                    src_frame_ids_temporal[1]
                    + 1
                    + self.n_src_views_temporal_track_one_side,
                ),
            ).tolist()
            n_actual_temporal_track_bwd2tgt = len(tmp_list)
            for i in range(n_actual_temporal_track_bwd2tgt):
                if tmp_list[i] in self.train_info_dict[scene_id]["time_ids"]:
                    src_frame_ids_temporal_track_bwd2tgt[i] = tmp_list[i]

        raw_c2w_tgt = np.linalg.inv(tgt_w2c)

        if self.spatial_src_view_type == "closest_wo_temporal":
            sorted_idxs_spatial, sorted_cam_dists_spatial = self.sort_poses_wrt_ref(
                tgt_pose=raw_c2w_tgt,
                ref_poses=np.copy(self.train_info_dict[scene_id]["train_c2w"]),
                tgt_id=-1,  # no need to mask tgt frame
                dist_method="dist_matrix",
            )  # [#frames, ]

            src_frame_ids_spatial = [
                self.train_info_dict[scene_id]["time_ids"][_]
                for _ in sorted_idxs_spatial
            ]

            src_frame_ids_spatial = src_frame_ids_spatial[: self.n_src_views_spatial]
        elif self.spatial_src_view_type == "closest_with_temporal":
            time_id_dist = np.abs(
                np.array(self.train_info_dict[scene_id]["time_ids"]).astype(np.float32)
                - float(tgt_time_id)
            )
            sorted_frame_ids = np.argsort(time_id_dist)[
                : (self.n_src_views_spatial * 4)
            ].tolist()
            src_frame_ids_spatial_pool = self.train_info_dict[scene_id]["time_ids"][
                sorted_frame_ids
            ]

            # print("\nsrc_frame_ids_spatial_pool: ", src_frame_ids_spatial_pool, "\n")

            c2w_pool = [_ - min_train_time_id for _ in src_frame_ids_spatial_pool]

            sorted_idxs_spatial, sorted_cam_dists_spatial = self.sort_poses_wrt_ref(
                tgt_pose=raw_c2w_tgt,
                ref_poses=np.copy(
                    self.train_info_dict[scene_id]["train_c2w"][c2w_pool, ...]
                ),
                tgt_id=-1,  # no need to mask tgt frame
                dist_method="dist_matrix",
            )  # [#frames, ]

            src_frame_ids_spatial = src_frame_ids_spatial_pool[sorted_idxs_spatial][
                : self.n_src_views_spatial
            ]
        elif self.spatial_src_view_type == "clustered":
            kmeans = KMeans(
                n_clusters=self.n_src_views_spatial_cluster,
                random_state=0,
                n_init="auto",
            ).fit(self.train_info_dict[scene_id]["train_c2w"][:, :3, 3])
            # kmeans.predict([[0, 0], [12, 3]])

            cluster_centers = kmeans.cluster_centers_  # [#cluster, 3]
            cluster_labels = kmeans.labels_  # [#frame,]
            unique_labels = list(set(cluster_labels.tolist()))

            assert (
                len(unique_labels) == self.n_src_views_spatial_cluster
            ), f"{len(unique_labels)}, {self.n_src_views_spatial_cluster}"

            # choose spatially closest clusters
            cluster_tgt_dist = np.linalg.norm(
                cluster_centers - raw_c2w_tgt[:3, 3].reshape((1, 3)), axis=1
            )  # [#cluster,]
            sorted_cluster_labels = np.argsort(cluster_tgt_dist)

            src_frame_ids_spatial = []
            for tmp_label in sorted_cluster_labels[: self.n_src_views_spatial]:
                tmp_cluster_elem_ids = np.nonzero(cluster_labels == tmp_label)[
                    0
                ]  # [#elem]

                # we find the temporally closest
                time_id_dist = np.abs(
                    tmp_cluster_elem_ids.astype(np.float32) - float(tgt_time_id)
                )
                sorted_frame_ids_temporally = np.argsort(time_id_dist).tolist()

                src_frame_ids_spatial.append(
                    tmp_cluster_elem_ids[sorted_frame_ids_temporally[0]]
                )
        else:
            raise ValueError(self.spatial_src_view_type)

        assert len(set(src_frame_ids_spatial)) == len(
            src_frame_ids_spatial
        ), f"{src_frame_ids_spatial}"
        src_frame_ids_spatial = sorted(src_frame_ids_spatial)

        frame_ids = np.array(
            [tgt_time_id, *src_frame_ids_spatial, *src_frame_ids_temporal]
        ).astype(int)

        # process RGB
        tgt_h, tgt_w, _ = tgt_rgb.shape
        tgt_shape = (tgt_h, tgt_w)

        covis_mask = self.parser_dict[scene_id].load_covisible(
            tgt_time_id, tgt_cam_id, "val"
        )  # True indicates co-visible areas
        covis_mask = (covis_mask > 0).astype(np.float32)  # [H, W]
        assert covis_mask.shape[0] == tgt_h, f"{covis_mask.shape[0]}, {tgt_h}"
        assert covis_mask.shape[1] == tgt_w, f"{covis_mask.shape[1]}, {tgt_w}"

        rgb_tgt, _, flat_cam_tgt, aug_info, _ = self._process_for_single_src_view(
            scene_id=scene_id,
            time_id=tgt_time_id,
            cam_id=tgt_cam_id,
            view_in_train=False,
            tgt_shape=tgt_shape,
            aug_type=None,
            square_pad_info=None,
            crop_info=None,
            use_dyn_mask=False,
            use_depth=False,
        )

        cur_aug_type = aug_info["aug_type"]
        cur_square_pad_info = aug_info["pad_info"]

        assert cur_aug_type == "none", f"{cur_aug_type}"

        (
            rgb_src_spatial,
            flat_cam_src_spatial,
            dyn_rgb_src_spatial,
            static_rgb_src_spatial,
            dyn_mask_src_spatial,
            depth_src_spatial,
            pcl_src_spatial,
        ) = self._extract_information(
            time_id_list=src_frame_ids_spatial,
            scene_id=scene_id,
            cam_id=None,
            view_in_train=True,
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
            np.linalg.inv(raw_c2w_tgt), coords_world_homo.T
        ).T  # [#pt, 4]

        # originally 0.1, 0.9
        depth_range_min = max(
            self.parser_dict[scene_id].near, np.quantile(coords_cam_tgt[:, 2], 0.1)
        )
        depth_range_max = min(
            self.parser_dict[scene_id].far, np.quantile(coords_cam_tgt[:, 2], 0.9)
        )

        depth_range = np.array([depth_range_min, depth_range_max])

        depth_range = np.tile(
            depth_range.reshape((1, 1, 2)), (tgt_h, tgt_w, 1)
        )  # [H, W, 2]

        # when there exists static content, we just set near/far to center around that depth
        flat_static_mask_src_spatial = (
            dyn_mask_src_spatial.reshape((-1)) == 0
        )  # [NxHxW]
        assert (
            flat_static_mask_src_spatial.shape[0] == pcl_src_spatial.shape[0]
        ), f"{flat_static_mask_src_spatial.shape}, {pcl_src_spatial.shape}"

        if np.sum(flat_static_mask_src_spatial) > 0:
            static_pcl_spatial = pcl_src_spatial[
                flat_static_mask_src_spatial, :
            ]  # [#staic, 3]

            # project to target
            tgt_K = flat_cam_tgt[2:18].reshape((4, 4))
            tgt_c2w = flat_cam_tgt[18:34].reshape((4, 4))
            static_pcl_spatial_homo = np.pad(
                static_pcl_spatial, ((0, 0), (0, 1)), mode="constant", constant_values=1
            )
            static_pcl_spatial_tgt_cam = np.matmul(
                np.linalg.inv(tgt_c2w), static_pcl_spatial_homo.T
            ).T[
                :, :3
            ]  # [#static, 3]
            static_depth = static_pcl_spatial_tgt_cam[:, 2]

            static_pix_coords = np.matmul(tgt_K[:3, :3], static_pcl_spatial_tgt_cam.T).T
            static_pix_coords = static_pix_coords[:, :2] / (
                static_pix_coords[:, 2:] + 1e-8
            )
            static_row_valid = (static_pix_coords[:, 1] >= 0) & (
                static_pix_coords[:, 1] <= tgt_h - 1
            )
            static_col_valid = (static_pix_coords[:, 0] >= 0) & (
                static_pix_coords[:, 0] <= tgt_w - 1
            )
            static_proj_valid = static_row_valid & static_col_valid

            if np.sum(static_proj_valid) > 0:
                static_pix_coords = static_pix_coords[static_proj_valid, :].astype(int)
                static_pix_coords_int = np.round(static_pix_coords).astype(int)

                static_valid_depth = static_depth[static_proj_valid]
                depth_range[
                    static_pix_coords_int[:, 1], static_pix_coords_int[:, 0], 0
                ] = (static_valid_depth - 1e-4)
                depth_range[
                    static_pix_coords_int[:, 1], static_pix_coords_int[:, 0], 1
                ] = (static_valid_depth + 1e-4)

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
            time_id_list=src_frame_ids_temporal,
            scene_id=scene_id,
            cam_id=None,
            view_in_train=True,
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
            time_id_list=src_frame_ids_temporal_track_fwd2tgt,
            scene_id=scene_id,
            cam_id=None,
            view_in_train=True,
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
            time_id_list=src_frame_ids_temporal_track_bwd2tgt,
            scene_id=scene_id,
            cam_id=None,
            view_in_train=True,
            tgt_shape=tgt_shape,
            aug_type=cur_aug_type,
            square_pad_info=cur_square_pad_info,
            use_dyn_mask=True,
            use_depth=True,
        )

        # read flow
        flow_fwd, flow_fwd_occ_mask = self._read_flow(
            scene_id=scene_id,
            src_time_id=src_frame_ids_temporal[0],
            tgt_time_id=src_frame_ids_temporal[1],
            tgt_shape=tgt_shape,
            occ_thres=self.flow_consist_thres,
            aug_type=cur_aug_type,
            square_pad_info=cur_square_pad_info,
            crop_info=None,
        )

        flow_bwd, flow_bwd_occ_mask = self._read_flow(
            scene_id=scene_id,
            src_time_id=src_frame_ids_temporal[1],
            tgt_time_id=src_frame_ids_temporal[0],
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
            # rgb tgt
            "rgb_tgt": torch.FloatTensor(rgb_tgt),  # [H, W, 3]
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
            "eval_mask": torch.FloatTensor(covis_mask)[..., None],  # [H, W, 1]
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
            "depth_src_spatial": torch.FloatTensor(depth_src_spatial)[..., None],  # [N, H, W, 1]
            "depth_src_temporal": torch.FloatTensor(depth_src_temporal)[..., None],  # [2, H, W, 1]
            "depth_src_temporal_track_fwd2tgt": torch.FloatTensor(depth_src_temporal_track_fwd2tgt)[..., None],  # [N, H, W, 1]
            "depth_src_temporal_track_bwd2tgt": torch.FloatTensor(depth_src_temporal_track_bwd2tgt)[..., None],  # [N, H, W, 1]
            "depth_range": torch.FloatTensor(depth_range),
            # time
            "time_tgt": torch.FloatTensor([tgt_time_id]),  # [1, ]
            "time_src_temporal": torch.FloatTensor(src_frame_ids_temporal),  # [2, ]
            "time_src_temporal_track_fwd2tgt": torch.FloatTensor(src_frame_ids_temporal_track_fwd2tgt),  # [N, ]
            "time_src_temporal_track_bwd2tgt": torch.FloatTensor(src_frame_ids_temporal_track_bwd2tgt),  # [N, ]
            # misc
            "misc": {
                "scene_id": scene_id,
                "tgt_frame_id": tgt_time_id,
                "tgt_cam_id": tgt_cam_id,
                "tgt_frame_name": tgt_frame_name,
            },
        }
        # fmt: on

        return ret_dict

    def _extract_view_info(self, parser, time_id, camera_id, view_in_train=True):
        rgb = parser.load_rgba(time_id, camera_id)[..., :3]  # [H, W, 3], uint8

        if view_in_train:
            depth = parser.load_depth(time_id, camera_id)[..., 0]  # [H, W, 1] -> [H, W]
        else:
            depth = None

        cam = parser.load_camera(time_id, camera_id)  # [H, W, 1]
        K = cam.intrin  # [3, 3]
        w2c = cam.extrin  # [4, 4]

        return rgb, depth, K, w2c

    def _get_img_f_for_src_view(self, scene_id, src_frame_id):
        # NOTE: for DynIBaR, the monocular video is selected from the original multiview vidoes in a round manner.
        # Namely, for the i-th frame, DynIBaR chooses the (i % 12)-th camera.
        # Therefore, for the monocular video, the frame and the camera has the same index.
        # Ref: https://github.com/google/dynibar/blob/1b15d7fb926cb997803514a344ce0a4c9086d49c/eval_nvidia.py#L317
        tmp_img_f = self.scene_img_dict[scene_id][src_frame_id][src_frame_id % N_CAMS]
        return tmp_img_f

    def _extract_information(
        self,
        *,
        time_id_list,
        scene_id,
        cam_id,
        view_in_train,
        tgt_shape,
        aug_type,
        square_pad_info,
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

        if view_in_train:
            assert cam_id is None, f"{cam_id}"
            cam_id = self.get_train_cam_id(scene_id)
        else:
            assert cam_id is not None

        for tmp_time_id in time_id_list:
            (
                tmp_rgb_src,
                _,
                tmp_flat_cam_src,
                tmp_aug_info,
                tmp_extra_info,
            ) = self._process_for_single_src_view(
                scene_id=scene_id,
                time_id=tmp_time_id,
                cam_id=cam_id,
                view_in_train=view_in_train,
                tgt_shape=tgt_shape,
                aug_type=aug_type,
                square_pad_info=square_pad_info,
                crop_info=None,
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
        scene_id,
        time_id,
        cam_id,
        view_in_train,
        tgt_shape,
        aug_type=None,
        square_pad_info=None,
        crop_info=None,
        use_dyn_mask=False,
        use_depth=False,
    ):
        need_crop = None
        assert crop_info is None, f"{crop_info}"

        raw_img, depth, K, w2c = self._extract_view_info(
            self.parser_dict[scene_id], time_id, cam_id, view_in_train=view_in_train
        )

        tgt_h, tgt_w = tgt_shape
        if raw_img.shape[0] != tgt_h or raw_img.shape[1] != tgt_w:
            raw_img = cv2.resize(
                raw_img,
                dsize=(tgt_w, tgt_h),
                interpolation=cv2.INTER_AREA,
            )

        rgb, rgb_mask, aug_info = self.process_img(
            raw_img,
            aug_type=aug_type,
            square_pad_info=square_pad_info,
            need_crop=need_crop,
            crop_info=crop_info,
        )

        if aug_type is not None:
            assert (
                aug_type == aug_info["aug_type"]
            ), f"{aug_type}, {aug_info['aug_type']}"

        assert aug_info["aug_type"] == "none", f'{aug_info["aug_type"]}'

        extra_info = {}

        if use_dyn_mask or use_depth:
            frame_name = self.parser_dict[scene_id].get_frame_name(time_id, cam_id)

            if use_dyn_mask:
                dyn_mask_f = (
                    self.mask_data_dir
                    / scene_id
                    / f"masks/final/{frame_name}_final.png"
                )

                dyn_mask = np.array(
                    PIL.Image.open(dyn_mask_f)
                )  # bool. True indicates dynamic part

                if dyn_mask.shape[0] != tgt_h or dyn_mask.shape[1] != tgt_w:
                    dyn_mask = np.array(
                        PIL.Image.fromarray(dyn_mask).resize(
                            (tgt_w, tgt_h), resample=PIL.Image.Resampling.NEAREST
                        )
                    )

                dyn_mask, _, _ = self.process_img(
                    dyn_mask[..., None],
                    aug_type=aug_info["aug_type"],
                    square_pad_info=aug_info["pad_info"],
                    need_crop=need_crop,
                    crop_info=aug_info["crop_info"],
                )

                dyn_mask = dyn_mask.astype(np.float32)[..., 0]  # [H, W]

                extra_info["dyn_mask"] = dyn_mask

            if use_depth:
                if depth.shape[0] != tgt_h or depth.shape[1] != tgt_w:
                    depth = cv2.resize(
                        depth, (tgt_w, tgt_h), interpolation=cv2.INTER_NEAREST
                    )

                depth, _, _ = self.process_img(
                    depth[..., None],
                    aug_type=aug_info["aug_type"],
                    square_pad_info=aug_info["pad_info"],
                    need_crop=need_crop,
                    crop_info=aug_info["crop_info"],
                )

                extra_info["depth"] = depth[..., 0]

        assert not self.use_normalized_K

        aug_c2w, aug_K = self._compute_cam_info(
            aug_type=aug_info["aug_type"],
            c2w=np.linalg.inv(w2c),
            K=K,
            tgt_shape=tgt_shape,
        )

        rgb = modify_rgb_range(rgb, src_range="0_255", tgt_range=self.rgb_range)

        if use_dyn_mask:
            # NOTE: we compute dyn_rgb here after rgb is converted to target range
            extra_info["dyn_rgb"] = rgb * dyn_mask[..., None].astype(np.float32)
            extra_info["static_rgb"] = rgb * (
                1 - dyn_mask[..., None].astype(np.float32)
            )

        cur_h, cur_w, _ = rgb.shape
        flat_cam_info = np.concatenate(
            ([cur_h, cur_w], aug_K.flatten(), aug_c2w.flatten())
        ).astype(
            np.float32
        )  # [34, ], 2 + 16 + 16

        if use_depth:
            pcl = self._compute_pcl(cur_h, cur_w, aug_K, aug_c2w, extra_info["depth"])
            extra_info["pcl"] = pcl

        return rgb, rgb_mask, flat_cam_info, aug_info, extra_info

    def _compute_pcl(self, h, w, K, c2w, depth):
        K_torch = torch.FloatTensor(K)[None, ...]
        c2w_torch = torch.FloatTensor(c2w)[None, ...]
        rays_o, rays_d, _, _ = self._get_rays_single_image(
            h, w, K_torch, c2w_torch
        )  # [H x W, 3]
        pcl = rays_o + rays_d * depth.reshape((-1, 1))
        return pcl

    def _compute_cam_info(self, *, aug_type, c2w, K, tgt_shape):
        aug_c2w, aug_K_3x3 = self.augment_cam(
            aug_type, c2w, K, tgt_shape[0], tgt_shape[1]
        )
        aug_K = np.eye(4)
        aug_K[:3, :3] = aug_K_3x3
        return aug_c2w, aug_K

    def _read_flow(
        self,
        *,
        scene_id,
        src_time_id,
        tgt_time_id,
        tgt_shape,
        aug_type,
        square_pad_info,
        crop_info,
        occ_thres=1,
    ):
        if src_time_id == tgt_time_id:
            aug_flow = np.zeros(list(tgt_shape) + [2], dtype=np.float32)
            aug_mask_occ = np.zeros(tgt_shape, dtype=np.float32)
        else:
            flow_interval = abs(tgt_time_id - src_time_id)

            train_cam_id = self.get_train_cam_id(scene_id)
            src_frame_name = self.parser_dict[scene_id].get_frame_name(
                src_time_id, train_cam_id
            )
            tgt_frame_name = self.parser_dict[scene_id].get_frame_name(
                tgt_time_id, train_cam_id
            )

            flow_f = (
                self.flow_data_dir
                / f"{scene_id}"
                / f"flows/interval_{flow_interval}"
                / f"{src_frame_name}_{tgt_frame_name}.npz"
            )
            flow_info = np.load(flow_f)

            flow = flow_info["flow"]  # [H, W, 2]
            coord_diff = flow_info["coord_diff"]  # [H, W, 2]
            mask_occ = (np.sum(np.abs(coord_diff), axis=2) > occ_thres).astype(
                np.float32
            )  # [H, W], bool

            assert flow.shape[0] == tgt_shape[0], f"{flow.shape}, {tgt_shape}"
            assert flow.shape[1] == tgt_shape[1], f"{flow.shape}, {tgt_shape}"

            need_crop = None

            aug_flow, _, _ = self.process_flow(
                flow,
                aug_type=aug_type,
                square_pad_info=square_pad_info,
                need_crop=need_crop,
                crop_info=crop_info,
            )

            aug_mask_occ, _, _ = self.process_img(
                mask_occ[..., None],
                aug_type=aug_type,
                square_pad_info=square_pad_info,
                need_crop=need_crop,
                crop_info=crop_info,
            )

            aug_mask_occ = aug_mask_occ[..., 0]

        return aug_flow, aug_mask_occ
