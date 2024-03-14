# modified from https://github.com/zhengqili/Neural-Scene-Flow-Fields/blob/d4001759a39b056c95d8bc22da34b10b4fb85afb/nsff_exp/load_llff.py
#
# Slightly modified version of LLFF data loading code
#  see https://github.com/Fyusion/LLFF for original


import io
import cv2
import logging
import zipfile
import pathlib
import PIL.Image
import numpy as np
from collections import defaultdict

import torch

from pgdvs.utils.rendering import modify_rgb_range
from pgdvs.utils.vis_utils import draw_cam_mesh
from pgdvs.datasets.base import BaseDataset


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

# fmt: off
ZOE_DEPTH_PRINCIPLE_DICT = {
    # mean absolute error
    "mae_med_share": ("disp_share_scale_med", "disp_share_shift_med"),
    "mae_med_indiv": ("disp_indiv_scale_med", "disp_indiv_shift_med"),
    "mae_trim_share": ("disp_share_scale_trim", "disp_share_shift_trim"),
    "mae_trim_indiv": ("disp_indiv_scale_trim", "disp_indiv_shift_trim"),
    # mean error
    "me_med_share": ("disp_share_scale_med", "disp_share_shift_med"),
    "me_med_indiv": ("disp_indiv_scale_med", "disp_indiv_shift_med"),
    "me_trim_share": ("disp_share_scale_trim", "disp_share_shift_trim"),
    "me_trim_indiv": ("disp_indiv_scale_trim", "disp_indiv_shift_trim"),
}
# fmt: on

N_CAMS = 12

TGT_HEIGHT = 288


LOGGER = logging.getLogger(__name__)


class NvidiaDynEvaluationDataset(BaseDataset):
    dataset_name = "NVIDIA_Dyn Eval"
    dataset_fname = "nvidia_eval"

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
        zoe_depth_data_path=None,
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
        self.n_src_views_temporal_track_one_side = n_src_views_temporal_track_one_side

        assert self.max_hw == -1, f"{self.max_hw}"

        self.use_normalized_K = False

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

        self.zoe_depth_data_path = None
        self.use_zoe_depth = use_zoe_depth
        if self.use_zoe_depth != "none":
            self.zoe_depth_data_path = pathlib.Path(data_root) / zoe_depth_data_path
            try:
                assert self.zoe_depth_data_path.exists(), self.zoe_depth_data_path
            except AssertionError:
                if self.zoe_depth_data_path.suffix in [".zip"]:
                    self.zoe_depth_data_path = (
                        self.zoe_depth_data_path.parent / self.zoe_depth_data_path.stem
                    )
                else:
                    self.zoe_depth_data_path = (
                        self.zoe_depth_data_path.parent
                        / f"{self.zoe_depth_data_path}.zip"
                    )
                assert self.zoe_depth_data_path.exists(), self.zoe_depth_data_path

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
            f"depth_data_dir: {self.depth_data_dir}; mask_data_dir: {self.mask_data_dir}; zoe_depth_data_path: {self.zoe_depth_data_path}.\n"
        )

        assert self.mode in ["eval"], f"{self.mode}"

        LOGGER.info(f"[{self.dataset_name}] [{self.mode}] use_aug: {use_aug}\n")

        LOGGER.info(f"[{self.dataset_name}] [{self.mode}]\n")

        if scene_ids is None:
            scene_ids = ALL_SCENE_IDS_NVIDIA_DYN

        LOGGER.info(
            f"[{self.dataset_name}] [{self.mode}] scene_ids: {scene_ids}, {type(scene_ids)}\n"
        )

        if self.mode == "train":
            raise NotImplementedError
        elif self.mode in ["eval"]:
            img_exts = PIL.Image.registered_extensions()
            supported_img_exts = {
                ex for ex, f in img_exts.items() if f in PIL.Image.OPEN
            }

            all_img_fs = []
            for tmp_ext in supported_img_exts:
                all_img_fs = all_img_fs + [
                    str(_.relative_to(self.raw_data_dir))
                    for _ in list(
                        self.raw_data_dir.glob(f"*/dense/mv_images/*/*{tmp_ext}")
                    )
                ]  # e.g., Balloon1/dense/mv_images/00000/cam01.jpg
            all_img_fs = sorted(all_img_fs)
        else:
            raise ValueError

        all_rgb_fs = []
        scene_img_dict = defaultdict(lambda: defaultdict(dict))
        for tmp in all_img_fs:
            if tmp.split("/")[0] in scene_ids:
                tmp_frame_id = int(pathlib.Path(tmp).parent.name)
                # camera starts from index 1, we make it to start from index 0
                tmp_cam_id = int(pathlib.Path(tmp).stem.split("cam")[1]) - 1
                tmp_scene_id = str(pathlib.Path(tmp).parent.parent.parent.parent)
                tmp_scene_dir = self.raw_data_dir / tmp_scene_id / "dense"

                scene_img_dict[tmp_scene_id][tmp_frame_id][tmp_cam_id] = str(
                    self.raw_data_dir / tmp
                )

                all_rgb_fs.append(
                    (
                        tmp_scene_id,
                        tmp_scene_dir,
                        tmp_frame_id,
                        tmp_cam_id,
                        str(self.raw_data_dir / tmp),
                    )
                )

        # convert to normal dict for pickling that is later used by multiple workers
        self.scene_img_dict = {k: dict(v) for k, v in scene_img_dict.items()}

        # NOTE: must sort here to ensure same order across workers
        self.valid_fs = sorted(list(set(all_rgb_fs)))

        LOGGER.info(
            f"[{self.dataset_name}] [{self.mode}] #final_valid_fs: {len(self.valid_fs)}\n"
        )

        self.zip_obj = None  # enable lazy read
        self.zoe_depth_zip_obj = None

    def __len__(self):
        return len(self.valid_fs)

    def _get_zip_obj(self):
        if self.zoe_depth_zip_obj is None and self.use_zoe_depth != "none":
            if self.zoe_depth_data_path.is_file():
                self.zoe_depth_zip_obj = zipfile.ZipFile(self.zoe_depth_data_path)

    def __getitem__(self, index):
        self._get_zip_obj()
        return self._get_item_func(index)

    def _get_item_func(self, index):
        scene_id, scene_dir, tgt_frame_id, tgt_cam_id, img_f = self.valid_fs[index]

        # True indicates that the current target frame is in the input monocular vidoe.
        flag_in_mono = tgt_frame_id % N_CAMS == tgt_cam_id

        all_hwf, all_c2w = self._read_cam(scene_id)
        n_img_fs = all_hwf.shape[0]

        src_frame_ids_temporal = []

        # temporally-closest frames
        if flag_in_mono:
            if tgt_frame_id > 0:
                older_frame_id = tgt_frame_id - 1
                src_frame_ids_temporal.append(older_frame_id)

            if tgt_frame_id < n_img_fs - 1:
                newer_frame_id = tgt_frame_id + 1
                src_frame_ids_temporal.append(newer_frame_id)
        else:
            # Since the target frame is not in the input video,
            # we can use the input frame at the same timestep.
            src_frame_ids_temporal.append(tgt_frame_id)

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
        if tgt_frame_id > 0:
            tmp_list = np.arange(
                max(
                    0,
                    src_frame_ids_temporal[0]
                    - self.n_src_views_temporal_track_one_side,
                ),
                src_frame_ids_temporal[0],
            ).tolist()
            src_frame_ids_temporal_track_fwd2tgt[: len(tmp_list)] = tmp_list
            n_actual_temporal_track_fwd2tgt = len(tmp_list)

        # newer frames
        n_actual_temporal_track_bwd2tgt = 0
        src_frame_ids_temporal_track_bwd2tgt = [
            src_frame_ids_temporal[1]
            for _ in range(self.n_src_views_temporal_track_one_side)
        ]
        if tgt_frame_id < n_img_fs - 1:
            tmp_list = np.arange(
                src_frame_ids_temporal[1] + 1,
                min(
                    n_img_fs,
                    src_frame_ids_temporal[1]
                    + 1
                    + self.n_src_views_temporal_track_one_side,
                ),
            ).tolist()
            src_frame_ids_temporal_track_bwd2tgt[: len(tmp_list)] = tmp_list
            n_actual_temporal_track_bwd2tgt = len(tmp_list)

        # NOTE: here use the camera ID !!!
        # For DynIBaR, poses are repeated. I.e., the first 12 cameras are repeated for the whole video.
        # Namely, for any i \in [0, 12], i and i + 12 x n has the same camera matrices for any n.
        raw_c2w_tgt = all_c2w[tgt_cam_id, ...]

        assert (
            self.n_src_views_spatial < N_CAMS * 2
        ), f"{self.n_src_views_spatial}, {N_CAMS * 2}"
        if flag_in_mono:
            # Since the target frame is in the monocular input video, we should not use it.
            src_frame_ids_spatial_pool = (
                np.arange(max(0, tgt_frame_id - N_CAMS), tgt_frame_id).tolist()
                + np.arange(
                    tgt_frame_id + 1, min(n_img_fs, tgt_frame_id + N_CAMS)
                ).tolist()
            )
            assert (
                tgt_frame_id not in src_frame_ids_spatial_pool
            ), f"{scene_id}, {tgt_frame_id}, {tgt_cam_id}"
        else:
            src_frame_ids_spatial_pool = np.arange(
                max(0, tgt_frame_id - N_CAMS), min(n_img_fs, tgt_frame_id + N_CAMS)
            ).tolist()

        sorted_frame_ids_spatial, sorted_cam_dists_spatial = self.sort_poses_wrt_ref(
            tgt_pose=raw_c2w_tgt,
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
            [tgt_frame_id, *src_frame_ids_spatial, *src_frame_ids_temporal]
        )

        # process RGB
        raw_rgb = np.array(PIL.Image.open(img_f))

        if raw_rgb.shape[0] != TGT_HEIGHT:
            mono_dir_list = list(pathlib.Path(scene_dir).glob(f"images_*x{TGT_HEIGHT}"))
            assert len(mono_dir_list) == 1, f"{mono_dir_list}"
            mono_dirname = mono_dir_list[0].name
            new_w, new_h = mono_dirname.split("images_")[1].split("x")
            new_h = int(new_h)
            new_w = int(new_w)
            raw_rgb = np.array(
                PIL.Image.fromarray(raw_rgb).resize(
                    (new_w, new_h), resample=PIL.Image.Resampling.LANCZOS
                )
            )

        raw_h, raw_w, _ = raw_rgb.shape

        assert raw_h == TGT_HEIGHT, f"{raw_h}, {TGT_HEIGHT}"

        # https://github.com/google/dynibar/blob/02b164144cce2d93aa4c5d87b418497286b2ae31/eval_nvidia.py#L423-L428
        eval_mask_dyn_f = (
            self.raw_data_dir
            / scene_id
            / f"dense/mv_masks/{tgt_frame_id:05d}"
            / f"cam{(tgt_cam_id + 1):02d}.png"
        )
        eval_mask_dyn = np.float32(
            cv2.imread(str(eval_mask_dyn_f)) > 1e-3
        )  # range [0, 1]
        eval_mask_dyn = cv2.resize(
            eval_mask_dyn,
            dsize=(raw_w, raw_h),
            interpolation=cv2.INTER_NEAREST,
        )  # [H, W, 3]

        # https://github.com/google/dynibar/blob/02b164144cce2d93aa4c5d87b418497286b2ae31/ibrnet/data_loaders/llff_data_utils.py#L107
        all_hwf[:, 0] = raw_h
        all_hwf[:, 1] = raw_w

        tgt_shape = (raw_h, raw_w)

        rgb_tgt, _, flat_cam_tgt, aug_info, _ = self._process_for_single_src_view(
            img_f=img_f,
            tgt_shape=tgt_shape,
            c2w=raw_c2w_tgt,
            hwf=all_hwf[tgt_cam_id, :],
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
            np.linalg.inv(raw_c2w_tgt), coords_world_homo.T
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
            "eval_mask": torch.FloatTensor(eval_mask_dyn),  # [H, W, 3]
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
            "time_tgt": torch.FloatTensor([tgt_frame_id]),  # [1, ]
            "time_src_temporal": torch.FloatTensor(src_frame_ids_temporal),  # [2, ]
            "time_src_temporal_track_fwd2tgt": torch.FloatTensor(src_frame_ids_temporal_track_fwd2tgt),  # [N, ]
            "time_src_temporal_track_bwd2tgt": torch.FloatTensor(src_frame_ids_temporal_track_bwd2tgt),  # [N, ]
            # misc
            "misc": {
                "scene_id": scene_id,
                "tgt_frame_id": tgt_frame_id,
                "tgt_cam_id": tgt_cam_id,
            },
        }
        # fmt: on

        return ret_dict

    def _read_cam(self, scene_id):
        scene_dir = self.raw_data_dir / scene_id / "dense"

        # NOTE: here we use poses_bounds_cvd because we use the DynIBaR-processed depths,
        # which is bonded to the poses_bounds_cvd.npy.
        cam_f = pathlib.Path(scene_dir) / "poses_bounds_cvd.npy"
        poses_arr = np.load(cam_f, allow_pickle=True)  # [#frames, 17]

        n_img_fs = poses_arr.shape[0]
        assert (
            len(self.scene_img_dict[scene_id].keys()) == n_img_fs
        ), f"{len(self.scene_img_dict[scene_id].keys())}, {n_img_fs}, {scene_id}"

        poses = (
            poses_arr[:, :-2].reshape([-1, 3, 5]).transpose([1, 2, 0])
        )  # [3, 5, #frame]
        bds = poses_arr[:, -2:].transpose([1, 0])  # [2, #frame]

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

        return all_hwf, all_c2w

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
        frame_id_list,
        scene_id,
        all_c2w,
        all_hwf,
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

        for tmp_src_id in frame_id_list:
            tmp_img_f = self._get_img_f_for_src_view(scene_id, tmp_src_id)
            (
                tmp_rgb_src,
                _,
                tmp_flat_cam_src,
                tmp_aug_info,
                tmp_extra_info,
            ) = self._process_for_single_src_view(
                img_f=tmp_img_f,
                tgt_shape=tgt_shape,
                c2w=all_c2w[tmp_src_id, ...],
                hwf=all_hwf[tmp_src_id, :],
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
        img_f,
        tgt_shape,
        c2w,
        hwf,
        aug_type=None,
        square_pad_info=None,
        crop_info=None,
        use_dyn_mask=False,
        use_depth=False,
    ):
        need_crop = None
        assert crop_info is None, f"{crop_info}"

        raw_img = np.array(PIL.Image.open(img_f))

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
            tmp_frame_id = int(pathlib.Path(img_f).parent.name)
            # camera starts from index 1, we make it to start from index 0
            tmp_cam_id = int(pathlib.Path(img_f).stem.split("cam")[1]) - 1
            tmp_scene_id = pathlib.Path(img_f).parent.parent.parent.parent.stem
            assert (
                tmp_cam_id == tmp_frame_id % N_CAMS
            ), f"{img_f}, {tmp_cam_id}, {tmp_frame_id}"

            if use_dyn_mask:
                dyn_mask = self._read_mask(tmp_scene_id, tmp_frame_id, tgt_h, tgt_w)

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
                depth = self._read_depth(tmp_scene_id, tmp_frame_id)  # [H, W]

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
            aug_type=aug_info["aug_type"], c2w=c2w, hwf=hwf, tgt_shape=tgt_shape
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

    def _read_mask(self, scene_id, frame_id, tgt_h, tgt_w):
        dyn_mask_f = (
            self.mask_data_dir
            / scene_id
            / f"dense/masks/final/{frame_id:05d}_final.png"
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

        return dyn_mask

    def _read_depth(self, scene_id, frame_id):
        if self.use_zoe_depth == "none":
            disp_f = self.depth_data_dir / f"{scene_id}/disp" / f"{frame_id:05d}.npy"
            depth = 1 / (np.load(disp_f) + 1e-8)  # [H, W]
        else:
            if self.use_zoe_depth == "moe":
                tmp_list = []
                for k in self.zoe_k_dict:
                    tmp_type, tmp_principle = self.zoe_k_dict[k]
                    if self.zoe_depth_zip_obj is None:
                        tmp_f = (
                            self.zoe_depth_data_path
                            / scene_id
                            / "dense"
                            / f"zoe_depths_{tmp_type}"
                            / f"{frame_id:05d}.npz"
                        )
                        tmp_info = np.load(tmp_f)
                        tmp_list.append(
                            (
                                tmp_type,
                                tmp_principle,
                                float(tmp_info[tmp_principle]),
                            )
                        )
                    else:
                        tmp_f = f"{self.zoe_depth_data_path.stem}/{scene_id}/dense/zoe_depths_{tmp_type}/{frame_id:05d}.npz"
                        with io.BufferedReader(
                            self.zoe_depth_zip_obj.open(tmp_f, mode="r")
                        ) as f:
                            tmp_info = np.load(f, allow_pickle=True)
                            tmp_list.append(
                                (
                                    tmp_type,
                                    tmp_principle,
                                    float(tmp_info[tmp_principle]),
                                )
                            )

                tmp_list = sorted(tmp_list, key=lambda x: abs(x[2]))
                zoe_best_type, zoe_best_principle, zoe_best_val = tmp_list[0]
            else:
                zoe_best_type, zoe_best_principle = self.zoe_k_dict[self.use_zoe_depth]

            if self.zoe_depth_zip_obj is None:
                best_f = (
                    self.zoe_depth_data_path
                    / scene_id
                    / "dense"
                    / f"zoe_depths_{zoe_best_type}"
                    / f"{frame_id:05d}.npz"
                )
                disp_info = np.load(best_f)

                pred_depth = disp_info["depth_pred"]
                disp_scale = disp_info[ZOE_DEPTH_PRINCIPLE_DICT[zoe_best_principle][0]]
                disp_shift = disp_info[ZOE_DEPTH_PRINCIPLE_DICT[zoe_best_principle][1]]
            else:
                best_f = f"{self.zoe_depth_data_path.stem}/{scene_id}/dense/zoe_depths_{zoe_best_type}/{frame_id:05d}.npz"
                with io.BufferedReader(
                    self.zoe_depth_zip_obj.open(best_f, mode="r")
                ) as f:
                    disp_info = np.load(f, allow_pickle=True)

                    pred_depth = disp_info["depth_pred"]
                    disp_scale = disp_info[
                        ZOE_DEPTH_PRINCIPLE_DICT[zoe_best_principle][0]
                    ]
                    disp_shift = disp_info[
                        ZOE_DEPTH_PRINCIPLE_DICT[zoe_best_principle][1]
                    ]

            raw_disp = 1.0 / (pred_depth + 1e-16)
            disp = disp_scale * raw_disp + disp_shift
            depth = 1 / (disp + 1e-16)

        return depth

    def _compute_cam_info(self, *, aug_type, c2w, hwf, tgt_shape):
        K = self._hwf_to_K(hwf, normalized=self.use_normalized_K, tgt_shape=tgt_shape)

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
        src_frame_id,
        tgt_frame_id,
        tgt_shape,
        aug_type,
        square_pad_info,
        crop_info,
        occ_thres=1,
    ):
        if src_frame_id == tgt_frame_id:
            aug_flow = np.zeros(list(tgt_shape) + [2], dtype=np.float32)
            aug_mask_occ = np.zeros(tgt_shape, dtype=np.float32)
        else:
            flow_interval = abs(tgt_frame_id - src_frame_id)
            flow_f = (
                self.flow_data_dir
                / f"{scene_id}"
                / f"dense/flows/interval_{flow_interval}"
                / f"{src_frame_id:05d}_{tgt_frame_id:05d}.npz"
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

    def _hwf_to_K(self, hwf, normalized=False, tgt_shape=None):
        h, w, f = hwf
        K = np.eye(3)
        K[0, 0] = f
        K[1, 1] = f
        K[0, 2] = w / 2.0
        K[1, 2] = h / 2.0

        if tgt_shape is not None:
            tgt_h, tgt_w = tgt_shape
            K[0, :] = K[0, :] * tgt_w / w
            K[1, :] = K[1, :] * tgt_h / h
            h, w = tgt_h, tgt_w

        if normalized:
            K[0, :] = K[0, :] / w
            K[1, :] = K[1, :] / h
        return K
