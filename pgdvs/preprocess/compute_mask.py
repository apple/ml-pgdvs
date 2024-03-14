# Modified based on:
# - https://github.com/SHI-Labs/OneFormer/blob/56799ef9e02968af4c7793b30deabcbeec29ffc0/demo/demo.py
# - https://github.com/CASIA-IVA-Lab/FastSAM/blob/1cb91e95160c9626d37bc07888c24eb44ad57a85/Inference.py
# - https://github.com/zhengqili/Neural-Scene-Flow-Fields/blob/d4001759a39b056c95d8bc22da34b10b4fb85afb/nsff_scripts/run_flows_video.py

import os
import sys
import copy
import time
import cv2
import random
import tqdm
import skimage
import pathlib
import argparse
import PIL.Image
import traceback
import imageio_ffmpeg
import numpy as np
import multiprocessing as mp
import matplotlib.pyplot as plt
from scipy import ndimage as ndi
from mpl_toolkits.axes_grid1 import make_axes_locatable

import torch

from detectron2.config import get_cfg
from detectron2.data.detection_utils import read_image
from detectron2.projects.deeplab import add_deeplab_config
from detectron2.utils.logger import setup_logger

from pgdvs.preprocess.common import read_poses_nvidia_long, hwf_to_K
from pgdvs.utils.rendering import images_to_video

cur_dir = pathlib.Path(__file__).parent.resolve()
oneformer_dir = cur_dir.parent.parent / "third_parties/OneFormer"
print("\n\noneformer_dir: ", oneformer_dir, "\n")
sys.path.insert(0, str(oneformer_dir))
oneformer_demo_dir = cur_dir.parent.parent / "third_parties/OneFormer/demo"
sys.path.insert(1, str(oneformer_demo_dir))

# for some reasons, "from third_parties.OneFormer.oneformer import XXX" will fail
from oneformer import (
    add_oneformer_config,
    add_common_config,
    add_swin_config,
    add_dinat_config,
    add_convnext_config,
)
from third_parties.OneFormer.demo.predictor import VisualizationDemo

FLAG_USE_FASTSAM = False

if FLAG_USE_FASTSAM:
    # performance degrades comparing to using SAM
    fastsam_dir = cur_dir.parent.parent.parent / "third_parties/FastSAM"
    sys.path.insert(0, str(fastsam_dir))
    print("\n\nfastsam_dir: ", fastsam_dir, "\n")

    # Stop YOLO/FastSAM's verbose output
    os.environ["YOLO_VERBOSE"] = "False"

    from third_parties.FastSAM.fastsam import FastSAM, FastSAMPrompt


# fmt: off
# Starting from index 1 !!!

# https://github.com/facebookresearch/detectron2/blob/67ac149947124670f6678e1bdd75f89dbf0dd5e7/detectron2/data/datasets/builtin_meta.py
# https://github.com/CSAILVision/sceneparsing/blob/e20222964c58eba41d8986d45d652d93adc234a5/objectInfo150.csv
DYNAMIC_IDS_ADE20K = [
    13,   # person
    21,   # car
    # 22,   # water
    # 42,   # box
    77,   # boat
    81,   # bus
    84,   # truck
    91,   # airplane
    93,   # dress/clothes
    103,  # van
    104,  # ship
    109,  # toy
    116,  # bag
    117,  # motorbike
    118,  # cradle
    120,  # ball
    127,  # animal
    128,  # bicycle
    140,  # fan
    150,  # flag
]

# NOTE: there are 133 categories in total for panoptic segmentation (https://cocodataset.org/#panoptic-eval)
# https://github.com/facebookresearch/detectron2/blob/67ac149947124670f6678e1bdd75f89dbf0dd5e7/detectron2/data/datasets/builtin_meta.py
# https://gist.github.com/AruniRC/7b3dadd004da04c80198557db5da4bda
# https://github.com/SHI-Labs/OneFormer/blob/4962ef6a96ffb76a76771bfa3e8b3587f209752b/oneformer/data/datasets/register_coco_panoptic_annos_semseg.py#L209
DYNAMIC_IDS_COCO = [
    1,   # person
    2,   # bicycle
    3,   # car
    4,   # motorcycle
    5,   # airplane
    6,   # bus
    7,   # train
    8,   # truck
    9,   # boat
    15,  # bird
    16,  # cat
    17,  # dog
    18,  # horse
    19,  # sheep
    20,  # cow
    21,  # elephant
    22,  # bear
    23,  # zebra
    24,  # giraffe
    # 25,  # backpack
    26,  # umbrella
    31,  # ski
    32,  # snowboard
    37,  # skateboard
    38,  # surfboard
    39,  # tennis racket
]
# fmt: on


FLAG_USE_WARP_PRE_MASK = True
FLAG_CONSIDER_PROB = True

SEM_SEG_PROB_THRES = 0.1


class MaskDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root=".",
    ):
        super(MaskDataset, self).__init__()
        self.image_root = pathlib.Path(root)

        exts = PIL.Image.registered_extensions()
        supported_exts = {ex for ex, f in exts.items() if f in PIL.Image.OPEN}

        image_list = []
        for tmp_ext in supported_exts:
            image_list = image_list + list(self.image_root.glob(f"*{tmp_ext}"))
        self.image_list = sorted(image_list)

        # print("\nimage_list: ", image_list, "\n")

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        return self.image_list[index]


def skew(x):
    return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])


def compute_epipolar_distance(*, T_12, K_1, K_2, p_1, p_2):
    R_12 = T_12[:3, :3]
    t_12 = T_12[:3, 3]

    E_mat = np.dot(skew(t_12), R_12)
    # compute bearing vector
    inv_K_1 = np.linalg.inv(K_1)
    inv_K_2 = np.linalg.inv(K_2)

    F_mat = np.dot(np.dot(inv_K_2.T, E_mat), inv_K_1)

    l_2 = np.dot(F_mat, p_1)
    algebric_e_distance = np.sum(p_2 * l_2, axis=0)
    n_term = np.sqrt(l_2[0, :] ** 2 + l_2[1, :] ** 2) + 1e-8
    geometric_e_distance = algebric_e_distance / n_term
    geometric_e_distance = np.abs(geometric_e_distance)

    return geometric_e_distance


def warp_flow(img, flow):
    h, w = flow.shape[:2]
    flow_new = flow.copy()
    flow_new[:, :, 0] += np.arange(w)
    flow_new[:, :, 1] += np.arange(h)[:, np.newaxis]

    res = cv2.remap(
        img, flow_new, None, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT
    )
    return res


def read_optical_flow(
    flow_dir, all_names, frame_idx, flow_interval=1, read_fwd=True, consist_thres=1.0
):
    if read_fwd:
        flow_fname = (
            f"{all_names[frame_idx]}_{all_names[frame_idx + flow_interval]}.npz"
        )
    else:
        flow_fname = (
            f"{all_names[frame_idx]}_{all_names[frame_idx - flow_interval]}.npz"
        )

    flow_info = np.load(flow_dir / flow_fname)
    flow = flow_info["flow"]  # [H, W, 2]
    coord_diff = flow_info["coord_diff"]  # [H, W, 2]

    # True values are for consistent pixels
    mask_consist = np.sum(np.abs(coord_diff), axis=2) <= consist_thres

    return flow, mask_consist


def compute_mask_epipolar_flow(
    *,
    img_ref,
    idx_ref,
    n_all_frames,
    all_w2c,
    all_K,
    flow_dir,
    flow_interval=1,
    threshold=1.0,
    all_img_names=[],
):
    img_ref_h, img_ref_w, _ = img_ref.shape

    xx = range(0, img_ref_w)
    yy = range(0, img_ref_h)  # , self.resized_h)
    xv, yv = np.meshgrid(xx, yy, indexing="xy")
    p_ref = np.float32(np.stack((xv, yv), axis=-1))
    p_ref_h = np.reshape(p_ref, (-1, 2))
    p_ref_h = np.concatenate((p_ref_h, np.ones((p_ref_h.shape[0], 1))), axis=-1).T

    w2c_ref = all_w2c[idx_ref, ...]
    K_ref = all_K[idx_ref, ...]

    # load optical flow
    if idx_ref < flow_interval:
        fwd_flow, fwd_mask = read_optical_flow(
            flow_dir, all_img_names, idx_ref, flow_interval=flow_interval, read_fwd=True
        )
        bwd_flow = np.zeros_like(fwd_flow)
        bwd_mask = np.zeros_like(fwd_mask)

        w2c_prev = np.eye(4)
        K_prev = np.eye(3)

        w2c_post = all_w2c[idx_ref + flow_interval, ...]
        K_post = all_K[idx_ref + flow_interval, ...]

        flag_use_prev = False

    elif idx_ref >= n_all_frames - flow_interval:
        bwd_flow, bwd_mask = read_optical_flow(
            flow_dir,
            all_img_names,
            idx_ref,
            flow_interval=flow_interval,
            read_fwd=False,
        )
        fwd_flow = np.zeros_like(bwd_flow)
        fwd_mask = np.zeros_like(bwd_mask)

        w2c_prev = all_w2c[idx_ref - flow_interval, ...]
        K_prev = all_K[idx_ref - flow_interval, ...]

        w2c_post = np.eye(4)
        K_post = np.eye(3)

        flag_use_prev = True

    else:
        fwd_flow, fwd_mask = read_optical_flow(
            flow_dir, all_img_names, idx_ref, flow_interval=flow_interval, read_fwd=True
        )
        bwd_flow, bwd_mask = read_optical_flow(
            flow_dir,
            all_img_names,
            idx_ref,
            flow_interval=flow_interval,
            read_fwd=False,
        )

        w2c_prev = all_w2c[idx_ref - flow_interval, ...]
        K_prev = all_K[idx_ref - flow_interval, ...]

        w2c_post = all_w2c[idx_ref + flow_interval, ...]
        K_post = all_K[idx_ref + flow_interval, ...]

        c2w_prev = np.linalg.inv(w2c_prev)
        c2w_ref = np.linalg.inv(w2c_ref)
        c2w_post = np.linalg.inv(w2c_post)

        dist_ref_prev = np.sum(np.abs(c2w_prev[:3, 3] - c2w_ref[:3, 3]))
        dist_ref_post = np.sum(np.abs(c2w_post[:3, 3] - c2w_ref[:3, 3]))

        # When cameras are too far away, the optical flow may not be reliable
        if dist_ref_prev < dist_ref_post:
            flag_use_prev = True
        else:
            flag_use_prev = False

    T_ref2prev = np.dot(w2c_prev, np.linalg.inv(w2c_ref))
    T_ref2post = np.dot(w2c_post, np.linalg.inv(w2c_ref))

    p_post = p_ref + fwd_flow
    p_post_h = np.reshape(p_post, (-1, 2))
    p_post_h = np.concatenate((p_post_h, np.ones((p_post_h.shape[0], 1))), axis=-1).T

    fwd_e_dist = compute_epipolar_distance(
        T_12=T_ref2post, K_1=K_ref, K_2=K_post, p_1=p_ref_h, p_2=p_post_h
    )
    fwd_e_dist = np.reshape(fwd_e_dist, (fwd_flow.shape[0], fwd_flow.shape[1]))

    p_prev = p_ref + bwd_flow
    p_prev_h = np.reshape(p_prev, (-1, 2))
    p_prev_h = np.concatenate((p_prev_h, np.ones((p_prev_h.shape[0], 1))), axis=-1).T

    bwd_e_dist = compute_epipolar_distance(
        T_12=T_ref2prev, K_1=K_ref, K_2=K_prev, p_1=p_ref_h, p_2=p_prev_h
    )
    bwd_e_dist = np.reshape(bwd_e_dist, (bwd_flow.shape[0], bwd_flow.shape[1]))

    if flag_use_prev:
        e_dist = bwd_e_dist * bwd_mask
    else:
        e_dist = fwd_e_dist * fwd_mask

    motion_mask = skimage.morphology.binary_opening(
        e_dist > threshold, skimage.morphology.disk(1)
    )

    return motion_mask


def combine_masks(
    *,
    mask_type,
    img_idx,
    sem_seg_ade20k,
    sem_seg_coco,
    mask_sam,
    mask_flow_epi,
    mask_flow_depth,
    prev_mask_final_raw=None,
    prev_dyn_cnt=None,
    normalized_dyn_track_thres=0.5,
    flow_dir=None,
    flow_interval=1,
    sam_overlap_thres=0.1,
    all_img_names=[],
):
    # Maybe using contour?
    # - https://github.com/facebookresearch/detectron2/blob/67ac149947124670f6678e1bdd75f89dbf0dd5e7/detectron2/utils/visualizer.py#L119
    # - https://github.com/facebookresearch/detectron2/blob/67ac149947124670f6678e1bdd75f89dbf0dd5e7/detectron2/utils/visualizer.py#L1035
    # - https://github.com/facebookresearch/detectron2/blob/67ac149947124670f6678e1bdd75f89dbf0dd5e7/detectron2/utils/visualizer.py#L1114

    mask_ade20k = None
    mask_coco = None
    mask_sem = None

    if mask_type == "semantic":
        mask_ade20k = np.zeros_like(sem_seg_ade20k, dtype=bool)
        for tmp_idx in DYNAMIC_IDS_ADE20K:
            # Class ID starts from 1. We need to change it to starting from 0
            tmp_rows, tmp_cols = np.nonzero(sem_seg_ade20k == (tmp_idx - 1))
            mask_ade20k[tmp_rows, tmp_cols] = True

        mask_coco = np.zeros_like(sem_seg_coco, dtype=bool)
        for tmp_idx in DYNAMIC_IDS_COCO:
            # Class ID starts from 1. We need to change it to starting from 0
            tmp_rows, tmp_cols = np.nonzero(sem_seg_coco == (tmp_idx - 1))
            mask_coco[tmp_rows, tmp_cols] = True

        mask_sem = mask_ade20k | mask_coco

        raw_mask_no_warp = mask_sem
    elif mask_type == "flow_epi":
        raw_mask_no_warp = mask_flow_epi
    elif mask_type == "flow_depth":
        raw_mask_no_warp = mask_flow_depth
    else:
        raise ValueError(mask_type)

    if prev_mask_final_raw is not None:
        bwd_flow, bwd_mask = read_optical_flow(
            flow_dir,
            all_img_names,
            img_idx,
            flow_interval=flow_interval,
            read_fwd=False,
        )
        bwd_mask = bwd_mask.astype(np.float32)

        mask_warp_prev_raw = warp_flow(prev_mask_final_raw.astype(np.uint8), bwd_flow)
        mask_warp_prev_raw = (mask_warp_prev_raw.astype(np.float32) * bwd_mask) > 1e-3

        # This tracks how many times a pixel is treated as dynamic.
        # If the pixel is mostly treated as static, then we do not consider it as a pixel for dynamic objects.
        dyn_cnt_warp_prev = warp_flow(prev_dyn_cnt, bwd_flow).astype(np.float32)

        mask_dyn_track = (
            dyn_cnt_warp_prev / (img_idx + 1) * bwd_mask
        ) > normalized_dyn_track_thres

        mask_warp_prev = mask_warp_prev_raw & mask_dyn_track

        # account for inaccuracies of the optical flow
        mask_warp_prev = skimage.morphology.binary_erosion(
            mask_warp_prev, skimage.morphology.disk(2)
        )

        raw_mask = raw_mask_no_warp | mask_warp_prev
    else:
        cur_dyn_cnt = raw_mask_no_warp.astype(np.float32)

        mask_warp_prev_raw = None
        mask_warp_prev = None
        mask_dyn_track = None
        raw_mask = raw_mask_no_warp

    raw_mask_eroded = skimage.morphology.binary_erosion(
        raw_mask, skimage.morphology.disk(2)
    )

    n_seg, _, _ = mask_sam.shape

    # mask_final_raw = np.zeros_like(raw_mask)
    mask_final_raw = np.copy(raw_mask_eroded)
    for tmp_idx in range(n_seg):
        tmp_seg = mask_sam[tmp_idx, ...]
        tmp_n_pixs = np.sum(tmp_seg.astype(float))
        tmp_n_overlap = np.sum((tmp_seg & raw_mask_eroded).astype(float))
        if (tmp_n_overlap > 0) and (tmp_n_overlap > sam_overlap_thres * tmp_n_pixs):
            # exist overlaps
            mask_final_raw[tmp_seg] = True

    if prev_mask_final_raw is not None:
        # TODO: should we use bwd_mask ???
        # cur_dyn_cnt = bwd_mask * dyn_cnt_warp_prev + mask_final_raw.astype(np.float32)
        cur_dyn_cnt = dyn_cnt_warp_prev + mask_final_raw.astype(np.float32)

    mask_final = (
        skimage.morphology.binary_dilation(
            # ndi.binary_fill_holes(mask_final_raw),
            mask_final_raw,
            skimage.morphology.disk(2),
        )
        > 1e-3
    )

    ret_dict = {
        "ade20k": mask_ade20k,
        "coco": mask_coco,
        "sem": mask_sem,
        "warp_prev": mask_warp_prev_raw,
        "dyn_track": mask_dyn_track,
        "dyn_cnt": cur_dyn_cnt,
        "raw_no_warp": raw_mask_no_warp,
        "raw": raw_mask,
        "raw_eroded": raw_mask_eroded,
        "final_raw": mask_final_raw,
        "final": mask_final,
    }

    return ret_dict


def setup_cfg_oneformer(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    add_deeplab_config(cfg)
    add_common_config(cfg)
    add_swin_config(cfg)
    add_dinat_config(cfg)
    add_convnext_config(cfg)
    add_oneformer_config(cfg)
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg


def show_sam_anns(raw_img, anns, save_f):
    # https://github.com/facebookresearch/segment-anything/blob/6fdee8f2727f4506cfbbe553e23b895e27956588/notebooks/automatic_mask_generator_example.ipynb
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x["area"]), reverse=True)

    fig = plt.figure(figsize=(20, 20), frameon=False)
    plt.imshow(raw_img)
    plt.axis("off")

    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones(
        (
            sorted_anns[0]["segmentation"].shape[0],
            sorted_anns[0]["segmentation"].shape[1],
            4,
        )
    )
    img[:, :, 3] = 0
    for ann in sorted_anns:
        m = ann["segmentation"]
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask

    img = (img * 255).astype(np.uint8)

    ax.imshow(img)
    ax.axis("off")
    plt.savefig(save_f, bbox_inches="tight", pad_inches=0)
    plt.close()


def show_dyn_cnt(dyn_cnt, save_f):
    # https://stackoverflow.com/a/18195921
    fig = plt.figure(figsize=(20, 20), frameon=False)
    ax = plt.gca()
    im = ax.imshow(dyn_cnt)
    ax.axis("off")
    # create an axes on the right side of ax. The width of cax will be 5%
    # of ax and the padding between cax and ax will be fixed at 0.05 inch.
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(im, cax=cax)
    plt.savefig(save_f, bbox_inches="tight", pad_inches=0)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default=".")
    parser.add_argument("--save_dir", type=str, default=".")
    parser.add_argument("--image_subdir", default="rgbs")
    parser.add_argument(
        "--mask_type",
        type=str,
        default="semantic",
        choices=["semantic", "flow_epi", "flow_depth"],
    )
    parser.add_argument("--oneformer_ckpt_dir", type=str, default=".")
    parser.add_argument("--sam_ckpt_f", type=str, default="sam_vit_h_4b8939.pth")
    parser.add_argument("--fastsam_ckpt_f", type=str, default="FastSAM-x.pt")
    parser.add_argument("--flow_interval", type=int, default=1)
    parser.add_argument("--flow_epi_thres", type=float, default=2.0)
    parser.add_argument("--for_colmap", action="store_true")
    parser.add_argument("--flag_dycheck_format", action="store_true")

    args = parser.parse_args()

    print("\nmask_type: ", args.mask_type, "\n")

    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:0")

    mp.set_start_method("spawn", force=True)
    setup_logger(name="fvcore")
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    oneformer_opts = [
        "MODEL.IS_TRAIN",
        "False",
        "MODEL.IS_DEMO",
        "True",
        "MODEL.WEIGHTS",
    ]

    # OneFormer ADE20K
    config_f_oneformer_ade20k = str(
        oneformer_dir / "configs/ade20k/dinat/oneformer_dinat_large_bs16_160k.yaml"
    )
    tmp_opts = copy.deepcopy(oneformer_opts)
    tmp_ckpt_f = str(
        pathlib.Path(args.oneformer_ckpt_dir)
        / "250_16_dinat_l_oneformer_ade20k_160k.pth"
    )
    tmp_opts.append(tmp_ckpt_f)
    tmp_args = argparse.Namespace(config_file=config_f_oneformer_ade20k, opts=tmp_opts)
    cfg_oneformer_ade20k = setup_cfg_oneformer(tmp_args)
    oneformer_ade20k = VisualizationDemo(cfg_oneformer_ade20k)

    # OneFormer COCO
    config_f_oneformer_coco = str(
        oneformer_dir / "configs/coco/dinat/oneformer_dinat_large_bs16_100ep.yaml"
    )
    tmp_opts = copy.deepcopy(oneformer_opts)
    tmp_ckpt_f = str(
        pathlib.Path(args.oneformer_ckpt_dir)
        / "150_16_dinat_l_oneformer_coco_100ep.pth"
    )
    tmp_opts.append(tmp_ckpt_f)
    tmp_args = argparse.Namespace(config_file=config_f_oneformer_coco, opts=tmp_opts)
    cfg_oneformer_coco = setup_cfg_oneformer(tmp_args)
    oneformer_coco = VisualizationDemo(cfg_oneformer_coco)

    oneformer_task = "semantic"

    if FLAG_USE_FASTSAM:
        # FastSAM
        model_fast_sam = FastSAM(args.fastsam_ckpt_f)
        fastsam_kwargs = {"retina_masks": True, "imgsz": 1024, "conf": 0.4, "iou": 0.9}
    else:
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

        sam = sam_model_registry["vit_h"](checkpoint=args.sam_ckpt_f).to(device)
        sam_mask_generator = SamAutomaticMaskGenerator(sam)

    root_dir = pathlib.Path(args.root_dir)

    data_dir_list = list(root_dir.glob(args.image_subdir))
    assert len(data_dir_list) == 1, f"{data_dir_list}"
    data_dir = data_dir_list[0]

    if args.for_colmap:
        flow_dirname = "flows_for_colmap"
    else:
        flow_dirname = "flows"
    flow_dir = root_dir / f"{flow_dirname}/interval_{args.flow_interval}"

    save_dir = pathlib.Path(args.save_dir)
    if args.for_colmap:
        out_dir = save_dir / "masks_for_colmap"
    else:
        out_dir = save_dir / "masks"
    out_dir.mkdir(exist_ok=True, parents=True)

    print("\nroot_dir: ", root_dir, save_dir, "\n")

    if FLAG_USE_FASTSAM:
        out_dir_sam = out_dir / "fastsam"
    else:
        out_dir_sam = out_dir / "sam"
    out_dir_sam.mkdir(exist_ok=True, parents=True)

    if args.mask_type == "semantic":
        out_dir_ade20k = out_dir / "ade20k"
        out_dir_ade20k.mkdir(exist_ok=True, parents=True)

        out_dir_coco = out_dir / "coco"
        out_dir_coco.mkdir(exist_ok=True, parents=True)
    elif args.mask_type == "flow_epi":
        out_dir_flow_epi = out_dir / "flow_epi"
        out_dir_flow_epi.mkdir(exist_ok=True, parents=True)
    elif args.mask_type == "flow_depth":
        out_dir_flow_depth = out_dir / "flow_depth"
        out_dir_flow_depth.mkdir(exist_ok=True, parents=True)
    else:
        raise ValueError(args.mask_type)

    out_dir_final = out_dir / "final"
    out_dir_final.mkdir(exist_ok=True, parents=True)

    val_dataset = MaskDataset(root=data_dir)

    n_all_frames = len(val_dataset)
    print("\nval_dataset: ", n_all_frames, "\n")

    if args.mask_type in ["flow_epi", "flow_depth"]:
        if args.flag_dycheck_format:
            cam_f = root_dir / "camera.npz"
            cam_info = np.load(cam_f)
            all_K = cam_info["all_K"]
            all_w2c = cam_info["all_w2c"]
            print("\nall_K: ", all_K.shape, all_w2c.shape, "\n")
        else:
            cam_f = root_dir / "poses_bounds_cvd.npy"
            all_c2w, all_hwf = read_poses_nvidia_long(
                cam_f, n_all_frames
            )  # c2w: [#frame, 4, 4]; hwf: [#frame, 3]

            all_w2c = np.array(
                [np.linalg.inv(all_c2w[_, ...]) for _ in range(n_all_frames)]
            )  # [#frame, 4, 4]

            tmp_img = np.array(PIL.Image.open(val_dataset[0]))
            tmp_h, tmp_w, _ = tmp_img.shape

            # https://github.com/google/dynibar/blob/02b164144cce2d93aa4c5d87b418497286b2ae31/ibrnet/data_loaders/llff_data_utils.py#L107
            all_hwf[:, 0] = tmp_h
            all_hwf[:, 1] = tmp_w

            tmp_shape = (tmp_h, tmp_w)
            all_K = np.array(
                [
                    hwf_to_K(all_hwf[_, :], tgt_shape=tmp_shape, normalized=False)
                    for _ in range(n_all_frames)
                ]
            )  # [#frames, 3, 3]

    prev_mask_final_raw = None
    prev_dyn_cnt = None

    all_img_names = []
    for val_id in tqdm.tqdm(range(n_all_frames), disable=True):
        all_img_names.append(val_dataset[val_id].stem)

    for val_id in tqdm.tqdm(range(n_all_frames), desc="#frames"):
        img_f = val_dataset[val_id]

        img = read_image(img_f, format="BGR")  # np.ndarray, [H, W, 3], uint8

        sem_pred_ade20k = None
        sem_pred_coco = None
        mask_flow_epi = None
        mask_flow_depth = None

        if args.mask_type == "semantic":
            # ADE20K
            pred_ade20k, vis_output_ade20k = oneformer_ade20k.run_on_image(
                np.copy(img), oneformer_task
            )
            sem_pred_ade20k = (
                pred_ade20k["sem_seg"].argmax(dim=0).cpu().numpy()
            )  # [#class, H, W] -> [H, W], int64
            if FLAG_CONSIDER_PROB:
                sem_pred_ade20k_prob_invalid = (
                    np.max(pred_ade20k["sem_seg"].cpu().numpy(), axis=0)
                    < SEM_SEG_PROB_THRES
                )
                ade20k_prob_invalid_rows, ade20k_prob_invalid_cols = np.nonzero(
                    sem_pred_ade20k_prob_invalid
                )
                sem_pred_ade20k[ade20k_prob_invalid_rows, ade20k_prob_invalid_cols] = -1

            vis_output_ade20k["semantic_inference"].save(
                out_dir_ade20k / f"{img_f.stem}.png"
            )

            # COCO
            # pred_coco = oneformer_coco.predictor(img, oneformer_task)
            pred_coco, vis_output_coco = oneformer_coco.run_on_image(
                np.copy(img), oneformer_task
            )
            sem_pred_coco = (
                pred_coco["sem_seg"].argmax(dim=0).cpu().numpy()
            )  # [#class, H, W] -> [H, W], int64
            if FLAG_CONSIDER_PROB:
                sem_pred_coco_prob_invalid = (
                    np.max(pred_coco["sem_seg"].cpu().numpy(), axis=0)
                    < SEM_SEG_PROB_THRES
                )
                coco_prob_invalid_rows, coco_prob_invalid_cols = np.nonzero(
                    sem_pred_coco_prob_invalid
                )
                sem_pred_coco[coco_prob_invalid_rows, coco_prob_invalid_cols] = -1

            vis_output_coco["semantic_inference"].save(
                out_dir_coco / f"{img_f.stem}.png"
            )
        elif args.mask_type == "flow_epi":
            # mask from epipolar + flow
            mask_flow_epi = compute_mask_epipolar_flow(
                img_ref=np.copy(img),
                idx_ref=val_id,
                n_all_frames=n_all_frames,
                all_w2c=all_w2c,
                all_K=all_K,
                flow_dir=flow_dir,
                flow_interval=args.flow_interval,
                all_img_names=all_img_names,
                threshold=args.flow_epi_thres,
            )
        elif args.mask_type == "flow_depth":
            raise NotImplementedError
        else:
            raise ValueError(args.mask_type)

        if FLAG_USE_FASTSAM:
            # FastSAM
            fastasm_everything_results = model_fast_sam(
                np.array(img), device=device, **fastsam_kwargs
            )
            prompt_process = FastSAMPrompt(
                str(img_f), fastasm_everything_results, device=device
            )
            ann_sam = (
                prompt_process.everything_prompt().bool().cpu().numpy()
            )  # [#ann, H, W]
            prompt_process.plot(
                annotations=ann_sam,
                output_path=str(out_dir_sam / f"{img_f.stem}.png"),
                bboxes=None,
                points=None,
                point_label=None,
                withContours=True,
                better_quality=True,
            )
        else:
            mask_sam = sam_mask_generator.generate(np.copy(img))

            ann_sam = np.stack([_["segmentation"] for _ in mask_sam], axis=0)

            show_sam_anns(img, mask_sam, out_dir_sam / f"{img_f.stem}.png")

        mask_dict = combine_masks(
            mask_type=args.mask_type,
            img_idx=val_id,
            sem_seg_ade20k=sem_pred_ade20k,
            sem_seg_coco=sem_pred_coco,
            mask_sam=ann_sam,
            mask_flow_epi=mask_flow_epi,
            mask_flow_depth=mask_flow_depth,
            prev_dyn_cnt=prev_dyn_cnt,
            prev_mask_final_raw=prev_mask_final_raw,
            flow_dir=flow_dir,
            flow_interval=args.flow_interval,
            all_img_names=all_img_names,
        )

        if FLAG_USE_WARP_PRE_MASK:
            prev_mask_final_raw = skimage.morphology.binary_erosion(
                mask_dict["final_raw"], skimage.morphology.disk(2)
            )
            prev_dyn_cnt = mask_dict["dyn_cnt"]

        # fmt: off
        if mask_flow_epi is not None:
            vis_mask_flow_epi = (mask_flow_epi * 255).astype(np.uint8)
            PIL.Image.fromarray(vis_mask_flow_epi).save(out_dir_flow_epi / f"{img_f.stem}.png")
        if mask_flow_depth is not None:
            vis_mask_flow_depth = (mask_flow_depth * 255).astype(np.uint8)
            PIL.Image.fromarray(vis_mask_flow_depth).save(out_dir_flow_depth / f"{img_f.stem}.png")
        if sem_pred_ade20k is not None:
            PIL.Image.fromarray(mask_dict["ade20k"]).save(out_dir_ade20k / f"{img_f.stem}_mask.png")
        if sem_pred_coco is not None:
            PIL.Image.fromarray(mask_dict["coco"]).save(out_dir_coco / f"{img_f.stem}_mask.png")

        if mask_dict["warp_prev"] is not None:
            PIL.Image.fromarray(mask_dict["raw_no_warp"]).save(out_dir_final / f"{img_f.stem}_0_raw_no_warp.png")
            PIL.Image.fromarray(mask_dict["warp_prev"]).save(out_dir_final / f"{img_f.stem}_0_warp_prev.png")
            PIL.Image.fromarray(mask_dict["dyn_track"]).save(out_dir_final / f"{img_f.stem}_0_dyn_track.png")
            show_dyn_cnt(prev_dyn_cnt, out_dir_final / f"{img_f.stem}_0_dyn_cnt.png")

        PIL.Image.fromarray(mask_dict["raw"]).save(out_dir_final / f"{img_f.stem}_1_raw.png")
        PIL.Image.fromarray(mask_dict["raw_eroded"]).save(out_dir_final / f"{img_f.stem}_2_raw_eroded.png")
        
        PIL.Image.fromarray(mask_dict["final_raw"]).save(out_dir_final / f"{img_f.stem}_3_final_raw.png")
        if args.for_colmap:
            # https://colmap.github.io/faq.html#mask-image-regions
            # Features will only be extracted from areas with mask values of 1
            mask_for_colmap = ~mask_dict["final"]
            PIL.Image.fromarray(mask_for_colmap).save(out_dir / f"{img_f.name}.png")
        else:
            PIL.Image.fromarray(mask_dict["final"]).save(out_dir_final / f"{img_f.stem}_final.png")
        # fmt: on

    try:
        print("Check FFMPEG exists.")
        imageio_ffmpeg.get_ffmpeg_exe()
        flag_ffmpeg_exe = True
    except RuntimeError:
        traceback.print_exc()
        err = sys.exc_info()[0]
        print(err)
        print(
            f"FFMPEG is not properly set therefore we do not automatically generate videos."
        )
        flag_ffmpeg_exe = False

    if flag_ffmpeg_exe:
        if args.for_colmap:
            mask_f_list = sorted(list(out_dir.glob("*.png")))
        else:
            mask_f_list = sorted(
                list(out_dir_final.glob("*_final.png"))
            )  # e.g., 00000_final
        mask_list = [
            np.array(PIL.Image.open(_)).astype(np.uint8) * 255 for _ in mask_f_list
        ]
        mask_list = [np.tile(_[..., None], (1, 1, 3)) for _ in mask_list]

        images_to_video(
            mask_list,
            out_dir,
            "mask",
            fps=5,
            quality=9,
            disable_tqdm=False,
        )
