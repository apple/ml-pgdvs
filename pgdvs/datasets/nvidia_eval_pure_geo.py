import io
import cv2
import tqdm
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
from pgdvs.datasets.nvidia_eval import NvidiaDynEvaluationDataset


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

LOGGER = logging.getLogger(__name__)


class NvidiaDynPureGeoEvaluationDataset(NvidiaDynEvaluationDataset):
    dataset_name = "NVIDIA_Dyn Pure Geometry Eval"
    dataset_fname = "nvidia_eval_pure_geo"

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

        self.use_zoe_depth = "none"

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
                ]
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

        # we need to compute static parts point cloud from the whole video
        self.st_pcl_dict = {}
        for tmp_scene_id in scene_ids:
            self.st_pcl_dict[tmp_scene_id] = self._aggregate_static_pcl(tmp_scene_id)

        LOGGER.info(
            f"[{self.dataset_name}] [{self.mode}] #final_valid_fs: {len(self.valid_fs)}\n"
        )

        self.zip_obj = None  # enable lazy read

    def __len__(self):
        return len(self.valid_fs)

    def _aggregate_static_pcl(self, scene_id):
        scene_dir = self.raw_data_dir / scene_id / "dense"
        mono_dir_list = list(pathlib.Path(scene_dir).glob(f"images_*x{TGT_HEIGHT}"))

        assert len(mono_dir_list) == 1, f"{mono_dir_list}"
        mono_dir = mono_dir_list[0]
        tgt_w, tgt_h = mono_dir.name.split("images_")[1].split("x")
        tgt_h = int(tgt_h)
        tgt_w = int(tgt_w)

        all_hwf, all_c2w = self._read_cam(scene_id)
        n_img_fs = all_hwf.shape[0]

        # https://github.com/google/dynibar/blob/02b164144cce2d93aa4c5d87b418497286b2ae31/ibrnet/data_loaders/llff_data_utils.py#L107
        all_hwf[:, 0] = tgt_h
        all_hwf[:, 1] = tgt_w

        st_pcl = np.zeros((0, 3), dtype=np.float32)
        st_rgb = np.zeros((0, 3), dtype=np.float32)

        for i in tqdm.tqdm(range(n_img_fs)):
            tmp_img_f = mono_dir / f"{i:05d}.png"
            tmp_img = np.array(PIL.Image.open(tmp_img_f))

            if (tmp_img.shape[0] != tgt_h) or (tmp_img.shape[1] != tgt_w):
                tmp_img = np.array(
                    PIL.Image.fromarray(tmp_img).resize(
                        (tgt_w, tgt_h), resample=PIL.Image.Resampling.LANCZOS
                    )
                )

            tmp_img = modify_rgb_range(tmp_img, src_range="0_255", tgt_range="0_1")

            tmp_K = self._hwf_to_K(all_hwf[i, :], normalized=False)
            tmp_c2w = all_c2w[i, ...]

            tmp_depth = self._read_depth(scene_id, i)

            tmp_pcl = self._compute_pcl(tgt_h, tgt_w, tmp_K, tmp_c2w, tmp_depth)

            tmp_dyn_mask = self._read_mask(scene_id, i, tgt_h, tgt_w).astype(bool)
            tmp_st_mask = (~tmp_dyn_mask).reshape(-1)

            if i > 0:
                # To avoid explosion of the number of points,
                # we only add points that are not covered by projection of the existing point cloud.
                tmp_proj_mask = self._compute_pcl_proj_mask(
                    h=tgt_h,
                    w=tgt_w,
                    pcl=st_pcl,
                    K=tmp_K,
                    w2c=np.linalg.inv(tmp_c2w),
                    dyn_mask=tmp_dyn_mask,
                )  # [hxw,]

                assert (
                    tmp_pcl.shape[0] == tmp_st_mask.shape[0]
                ), f"{tmp_pcl.shape}, {tmp_st_mask.shape}"
                assert (
                    tmp_pcl.shape[0] == tmp_proj_mask.shape[0]
                ), f"{tmp_pcl.shape}, {tmp_proj_mask.shape}"

                tmp_st_mask = tmp_st_mask & (~tmp_proj_mask)

            tmp_st_pcl = tmp_pcl[tmp_st_mask, :]
            tmp_st_rgb = tmp_img.reshape((-1, 3))[tmp_st_mask, :]

            st_pcl = np.concatenate((st_pcl, tmp_st_pcl), axis=0)
            st_rgb = np.concatenate((st_rgb, tmp_st_rgb), axis=0)

        st_pcl_rgb = np.concatenate((st_pcl, st_rgb), axis=1)

        return st_pcl_rgb

    def _compute_pcl_proj_mask(self, *, h, w, pcl, K, w2c, dyn_mask):
        verts_homo = np.pad(
            pcl, ((0, 0), (0, 1)), mode="constant", constant_values=1
        )  # [N, 3] -> [N, 4]

        verts_cam = np.matmul(w2c, verts_homo.T).T
        verts_cam = verts_cam[:, :3] / verts_cam[:, 3:]

        pix_coords = np.matmul(K[:3, :3], verts_cam.T).T  # [#pt, 3]

        pix_coords = pix_coords[..., :2] / pix_coords[..., 2:]

        row_valid = (pix_coords[:, 1] >= 0) & (pix_coords[:, 1] <= h - 1)
        col_valid = (pix_coords[:, 0] >= 0) & (pix_coords[:, 0] <= w - 1)

        pix_coords = pix_coords[row_valid & col_valid, :].astype(int)  # [#pt, 2]

        mask = np.zeros((h, w), dtype=bool)
        mask[pix_coords[:, 1], pix_coords[:, 0]] = True

        return mask.reshape(-1)

    def __getitem__(self, index):
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

        frame_ids = np.array([tgt_frame_id, *src_frame_ids_temporal])

        # NOTE: here use the camera ID !!!
        # For DynIBaR, poses are repeated. I.e., the first 12 cameras are repeated for the whole video.
        # Namely, for any i \in [0, 12], i and i + 12 x n has the same camera matrices for any n.
        raw_c2w_tgt = all_c2w[tgt_cam_id, ...]

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
            "st_pcl_rgb": torch.FloatTensor(self.st_pcl_dict[scene_id]),  # [#pt, 6], 1st 3 for points and 2nd 3 for RGB
            # rgb tgt
            "rgb_tgt": torch.FloatTensor(rgb_tgt),  # [H, W, 3]
            # rgb src temporal
            "n_actual_temporal": torch.LongTensor([n_actual_temporal]),  # [1,]
            "rgb_src_temporal": torch.FloatTensor(rgb_src_temporal),  # [2, H, W, 3]
            "dyn_rgb_src_temporal": torch.FloatTensor(dyn_rgb_src_temporal),  # [2, H, W, 3]
            "static_rgb_src_temporal": torch.FloatTensor(static_rgb_src_temporal),  # [2, H, W, 3]
            # mask
            "dyn_mask_src_temporal": torch.FloatTensor(dyn_mask_src_temporal)[..., None],  # [2, H, W, 1]
            "eval_mask": torch.FloatTensor(eval_mask_dyn),  # [H, W, 3]
            # flow
            "flow_fwd": torch.FloatTensor(flow_fwd),  # [H, W, 2]
            "flow_fwd_occ_mask": torch.FloatTensor(flow_fwd_occ_mask)[..., None],  # [H, W, 1]
            "flow_bwd": torch.FloatTensor(flow_bwd),  # [H, W, 2]
            "flow_bwd_occ_mask": torch.FloatTensor(flow_bwd_occ_mask)[..., None],  # [H, W, 1]
            # cam info
            "flat_cam_tgt": torch.FloatTensor(flat_cam_tgt),  # [34,]
            "flat_cam_src_temporal": torch.FloatTensor(flat_cam_src_temporal),  # [2, 34]
            # depth
            "depth_src_temporal": torch.FloatTensor(depth_src_temporal)[..., None],  # [2, H, W, 1]
            # time
            "time_tgt": torch.FloatTensor([tgt_frame_id]),  # [1, ]
            "time_src_temporal": torch.FloatTensor(src_frame_ids_temporal),  # [2, ]
            # misc
            "misc": {
                "scene_id": scene_id,
                "tgt_frame_id": tgt_frame_id,
                "tgt_cam_id": tgt_cam_id,
            },
        }
        # fmt: on

        return ret_dict
