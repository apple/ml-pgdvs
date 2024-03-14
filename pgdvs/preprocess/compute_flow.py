# modified from
# - https://github.com/princeton-vl/RAFT/blob/3fa0bb0a9c633ea0a9bb8a79c576b6785d4e6a02/demo.py
# - https://github.com/drinkingcoder/FlowFormer-Official/blob/21be9d720b24310d1e739d044e812034ab875b33/visualize_flow.py

import os
import sys
import cv2
import math
import tqdm
import shutil
import pathlib
import argparse
import PIL.Image
import numpy as np

import torch
import torch.nn.functional as F

from pgdvs.preprocess.common import viz, warp, compute_occlusion, flow_to_image


FLOWFORMER_TRAIN_SIZE = [432, 960]

FLAG_SAVE_DEBUG_INFO = True


class DiffFlowDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root=".",
        difference=1,
    ):
        super(DiffFlowDataset, self).__init__()
        self.image_root = pathlib.Path(root)

        # print("\nself.image_root: ", self.image_root, "\n")

        exts = PIL.Image.registered_extensions()
        supported_exts = {ex for ex, f in exts.items() if f in PIL.Image.OPEN}

        self.image_list = []
        self.extra_info = []

        image_list = []
        for tmp_ext in supported_exts:
            image_list = image_list + list(self.image_root.glob(f"*{tmp_ext}"))
        image_list = sorted(image_list)

        # print("\nimage_list: ", image_list, "\n")

        for i in range(0, len(image_list) - difference):
            self.image_list += [[image_list[i], image_list[i + difference]]]

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        return self.image_list[index]


def compute_grid_indices(image_shape, patch_size=FLOWFORMER_TRAIN_SIZE, min_overlap=20):
    if (
        min_overlap >= FLOWFORMER_TRAIN_SIZE[0]
        or min_overlap >= FLOWFORMER_TRAIN_SIZE[1]
    ):
        raise ValueError(
            f"Overlap should be less than size of patch (got {min_overlap}"
            f"for patch size {patch_size})."
        )
    if image_shape[0] == FLOWFORMER_TRAIN_SIZE[0]:
        hs = list(range(0, image_shape[0], FLOWFORMER_TRAIN_SIZE[0]))
    else:
        hs = list(range(0, image_shape[0], FLOWFORMER_TRAIN_SIZE[0] - min_overlap))
    if image_shape[1] == FLOWFORMER_TRAIN_SIZE[1]:
        ws = list(range(0, image_shape[1], FLOWFORMER_TRAIN_SIZE[1]))
    else:
        ws = list(range(0, image_shape[1], FLOWFORMER_TRAIN_SIZE[1] - min_overlap))

    # Make sure the final patch is flush with the image boundary
    hs[-1] = image_shape[0] - patch_size[0]
    ws[-1] = image_shape[1] - patch_size[1]
    return [(h, w) for h in hs for w in ws]


def compute_adaptive_image_size(image_size):
    target_size = FLOWFORMER_TRAIN_SIZE
    scale0 = target_size[0] / image_size[0]
    scale1 = target_size[1] / image_size[1]

    if scale0 > scale1:
        scale = scale0
    else:
        scale = scale1

    image_size = (int(image_size[1] * scale), int(image_size[0] * scale))

    return image_size, scale


def prepare_image_flowformer(fn1, fn2, flowformer_use_tile, max_hw=-1):
    ori_image1 = PIL.Image.open(fn1)
    ori_image2 = PIL.Image.open(fn2)

    if max_hw > 0:
        tmp_w, tmp_h = ori_image1.size
        max_hw_scale = max_hw / max(tmp_h, tmp_w)
        tmp_new_h = int(tmp_h * max_hw_scale)
        tmp_new_w = int(tmp_w * max_hw_scale)
        ori_image1 = ori_image1.resize(
            (tmp_new_w, tmp_new_h), resample=PIL.Image.Resampling.LANCZOS
        )
        ori_image2 = ori_image2.resize(
            (tmp_new_w, tmp_new_h), resample=PIL.Image.Resampling.LANCZOS
        )

    ori_image1 = np.array(ori_image1).astype(np.uint8)[..., :3]
    ori_image2 = np.array(ori_image2).astype(np.uint8)[..., :3]

    ori_size = ori_image1.shape[0:2]
    scale = 1.0
    if not flowformer_use_tile:
        dsize, scale = compute_adaptive_image_size(ori_image1.shape[0:2])
        image1 = cv2.resize(ori_image1, dsize=dsize, interpolation=cv2.INTER_CUBIC)
        image2 = cv2.resize(ori_image2, dsize=dsize, interpolation=cv2.INTER_CUBIC)
    else:
        image1 = ori_image1
        image2 = ori_image2

    ori_image1 = torch.from_numpy(ori_image1).permute(2, 0, 1).float()[None, ...]
    ori_image2 = torch.from_numpy(ori_image2).permute(2, 0, 1).float()[None, ...]

    image1 = torch.from_numpy(image1).permute(2, 0, 1).float()[None, ...]
    image2 = torch.from_numpy(image2).permute(2, 0, 1).float()[None, ...]

    return image1, image2, (ori_size, scale, ori_image1, ori_image2)


def compute_weight(
    device,
    hws,
    image_shape,
    patch_size=FLOWFORMER_TRAIN_SIZE,
    sigma=1.0,
    wtype="gaussian",
):
    patch_num = len(hws)
    h, w = torch.meshgrid(torch.arange(patch_size[0]), torch.arange(patch_size[1]))
    h, w = h / float(patch_size[0]), w / float(patch_size[1])
    c_h, c_w = 0.5, 0.5
    h, w = h - c_h, w - c_w
    weights_hw = (h**2 + w**2) ** 0.5 / sigma
    denorm = 1 / (sigma * math.sqrt(2 * math.pi))
    weights_hw = denorm * torch.exp(-0.5 * (weights_hw) ** 2)

    weights = torch.zeros(1, patch_num, *image_shape)
    for idx, (h, w) in enumerate(hws):
        weights[:, idx, h : h + patch_size[0], w : w + patch_size[1]] = weights_hw
    weights = weights.to(device)
    patch_weights = []
    for idx, (h, w) in enumerate(hws):
        patch_weights.append(
            weights[:, idx : idx + 1, h : h + patch_size[0], w : w + patch_size[1]]
        )

    return patch_weights


def compute_flow_flowformer(model, image1, image2, sigma, flowformer_use_tile=False):
    # Assume [B, C, H, W]
    image_size = image1.shape[2:]

    hws = compute_grid_indices(image_size)

    if not flowformer_use_tile:  # no tile
        padder = InputPadderFlowFormer(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        flow_pre, _ = model(image1, image2)

        flow_pre = padder.unpad(flow_pre)
        flow = flow_pre[0].permute(1, 2, 0).cpu().numpy()
    else:  # tile
        flows = 0
        flow_count = 0

        weights = compute_weight(
            image1.device, hws, image_size, FLOWFORMER_TRAIN_SIZE, sigma
        )

        for idx, (h, w) in enumerate(hws):
            image1_tile = image1[
                :, :, h : h + FLOWFORMER_TRAIN_SIZE[0], w : w + FLOWFORMER_TRAIN_SIZE[1]
            ]
            image2_tile = image2[
                :, :, h : h + FLOWFORMER_TRAIN_SIZE[0], w : w + FLOWFORMER_TRAIN_SIZE[1]
            ]
            flow_pre, _ = model(image1_tile, image2_tile)
            padding = (
                w,
                image_size[1] - w - FLOWFORMER_TRAIN_SIZE[1],
                h,
                image_size[0] - h - FLOWFORMER_TRAIN_SIZE[0],
                0,
                0,
            )
            flows += F.pad(flow_pre * weights[idx], padding)
            flow_count += F.pad(weights[idx], padding)

        flow_pre = flows / flow_count
        # flow = flow_pre[0].permute(1, 2, 0).cpu().numpy()

    return flow_pre


def load_image_raft(imfile):
    img = np.array(PIL.Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None]


def compute_flow_raft(device, model, img_f1, img_f2):
    image1 = load_image_raft(img_f1).to(device)
    image2 = load_image_raft(img_f2).to(device)
    padder = InputPadderRAFT(image1.shape)
    image1, image2 = padder.pad(image1, image2)

    # Sec. 4 of https://arxiv.org/abs/2003.12039
    # 32 for sintel and 24 for kitti
    raft_n_iters = 32
    _, flow12 = model(image1, image2, iters=raft_n_iters, test_mode=True)
    _, flow21 = model(image2, image1, iters=raft_n_iters, test_mode=True)

    flow12 = padder.unpad(flow12)
    flow21 = padder.unpad(flow21)
    image1 = padder.unpad(image1).cpu()
    image2 = padder.unpad(image2).cpu()

    return image1, image2, flow12, flow21


def build_model(model_to_use, pretrained_ckpt_f, device):
    print(f"building  model...")

    if model_to_use == "raft":
        raft_args = argparse.Namespace(
            **{"small": False, "mixed_precision": False, "alternate_corr": False}
        )
        model = torch.nn.DataParallel(RAFT(raft_args))
        model.load_state_dict(torch.load(pretrained_ckpt_f))
        model = model.module
    elif model_to_use == "flowformer":
        cfg = get_cfg()
        model = torch.nn.DataParallel(build_flowformer(cfg))
        model.load_state_dict(torch.load(pretrained_ckpt_f))
    else:
        raise ValueError(model_to_use)

    model.to(device)
    model.eval()

    return model


def _resize_torch(x, tgt_h, tgt_w):
    return torch.nn.functional.interpolate(
        x,
        size=(tgt_h, tgt_w),
        mode="bilinear",
        align_corners=True,
    )


@torch.no_grad()
def run(
    *,
    device,
    input_dir,
    out_dir,
    model,
    sigma=0.05,
    img_pair_max_diff=3,
    model_to_use="raft",
    flowformer_use_tile=False,
    max_hw=-1,
):
    print("\n[Flow] input_dir: ", input_dir, "\n")
    print("\n[Flow] out_dir: ", out_dir, "\n")

    os.makedirs(out_dir, exist_ok=True)

    model.eval()

    for cur_diff in tqdm.tqdm(np.arange(1, img_pair_max_diff + 1), desc="#intervals"):
        # fmt: off
        tmp_prefix = "interval"
        flowpath = pathlib.Path(out_dir) / f"{tmp_prefix}_{cur_diff}"
        flowpath.mkdir(parents=True, exist_ok=True)
        # fmt: on

        val_dataset = DiffFlowDataset(root=input_dir, difference=cur_diff)

        for val_id in tqdm.tqdm(range(len(val_dataset)), desc="#frames"):
            cur_fnames = val_dataset[val_id]
            fn1, fn2 = cur_fnames

            if model_to_use == "raft":
                image1, image2, flow12, flow21 = compute_flow_raft(
                    device, model, fn1, fn2
                )
            elif model_to_use == "flowformer":
                image1, image2, img_extra_info = prepare_image_flowformer(
                    fn1, fn2, flowformer_use_tile, max_hw=max_hw
                )
                image1, image2 = image1.to(device), image2.to(device)
                ori_size, scale, ori_image1, ori_image2 = img_extra_info

                flow12 = compute_flow_flowformer(
                    model, image1, image2, sigma, flowformer_use_tile
                )  # [1, 2, H, W], float32
                flow21 = compute_flow_flowformer(
                    model, image2, image1, sigma, flowformer_use_tile
                )

                if not flowformer_use_tile:
                    # NOTE: this means that the image has been resized.
                    # We need to resize it back.
                    flow12 = _resize_torch(flow12 / scale, ori_size[0], ori_size[1])
                    flow21 = _resize_torch(flow21 / scale, ori_size[0], ori_size[1])
                    image1 = ori_image1
                    image2 = ori_image2
            else:
                raise ValueError(model_to_use)

            # NOTE: pay attention to the order of input
            coord_diff_1, err_1 = compute_occlusion(
                image1, flow12, flow21, return_raw=True
            )  # coord_diff: [1, 2, H, W]; err: [1, H, W]; both are float32
            coord_diff_2, err_2 = compute_occlusion(
                image2, flow21, flow12, return_raw=True
            )

            flow12 = flow12.permute(0, 2, 3, 1)[0, ...].cpu().numpy()
            flow21 = flow21.permute(0, 2, 3, 1)[0, ...].cpu().numpy()

            coord_diff_1 = coord_diff_1.permute(0, 2, 3, 1)[0, ...].cpu().numpy()
            coord_diff_2 = coord_diff_2.permute(0, 2, 3, 1)[0, ...].cpu().numpy()

            err_1 = err_1.permute(1, 2, 0).cpu().numpy()
            err_2 = err_2.permute(1, 2, 0).cpu().numpy()

            stem_12 = f"{fn1.stem}_{fn2.stem}"
            np.savez(flowpath / f"{stem_12}.npz", flow=flow12, coord_diff=coord_diff_1)

            flow_img12 = flow_to_image(flow12)
            PIL.Image.fromarray(flow_img12).save(flowpath / f"{stem_12}.png")

            stem_21 = f"{fn2.stem}_{fn1.stem}"
            np.savez(flowpath / f"{stem_21}.npz", flow=flow21, coord_diff=coord_diff_2)

            flow_img21 = flow_to_image(flow21)
            PIL.Image.fromarray(flow_img21).save(flowpath / f"{stem_21}.png")

            # NOTE: DEBUG
            if FLAG_SAVE_DEBUG_INFO:
                tmp_flow12 = torch.FloatTensor(flow12)[None, ...].permute(0, 3, 1, 2)
                tmp_flow21 = torch.FloatTensor(flow21)[None, ...].permute(0, 3, 1, 2)
                # print("\nimage2: ", image2.shape, tmp_flow12.shape, "\n")
                img21 = warp(image2, tmp_flow12)
                img12 = warp(image1, tmp_flow21)
                tmp_flo12, tmp_flo21 = viz(
                    img1=image1,
                    img2=image2,
                    flo12=tmp_flow12,
                    flo21=tmp_flow21,
                    img12=img12,
                    img21=img21,
                    occ1=(err_1 > 1)[..., 0],
                    occ2=(err_2 > 1)[..., 0],
                    save_f=flowpath / f"debug_{stem_12}.png",
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", default=".")
    parser.add_argument("--save_dir", default=".")
    parser.add_argument("--image_subdir", default="rgbs")
    parser.add_argument("--image_pattern", default="*")
    parser.add_argument("--ckpt_f", default="sintel.pth")
    parser.add_argument("--img_pair_max_diff", type=int, default=1)
    parser.add_argument("--max_hw", type=int, default=-1)
    parser.add_argument("--for_colmap", action="store_true")
    parser.add_argument("--model_to_use", choices=["raft", "flowformer"])
    parser.add_argument(
        "--flowformer_use_tile",
        action="store_true",
        help="use tiling when running on a large-resolution image, "
        "or the image will be adaptively resized to match training size.",
    )

    args = parser.parse_args()

    device = torch.device("cuda:0")

    print("\n[Flow] model to use: ", args.model_to_use, "\n")

    # NOTE: since RAFT and FlowFormer's import path are not 'relative"
    # and they share many common path, we can only import one model at a time.
    if args.model_to_use == "raft":
        cur_dir = pathlib.Path(__file__).parent.resolve()
        raft_dir = cur_dir.parent.parent / "third_parties/RAFT/core"
        sys.path.insert(0, str(raft_dir))
        print("\nraft_dir: ", raft_dir, "\n")

        from third_parties.RAFT.core.raft import RAFT
        from third_parties.RAFT.core.utils.utils import InputPadder as InputPadderRAFT
    elif args.model_to_use == "flowformer":
        cur_dir = pathlib.Path(__file__).parent.resolve()
        flowformer_dir = cur_dir.parent.parent / "third_parties/FlowFormer/core"
        sys.path.insert(0, str(flowformer_dir))
        print("\nflowformer_dir: ", flowformer_dir, "\n")

        from third_parties.FlowFormer.configs.submission import get_cfg
        from third_parties.FlowFormer.core.FlowFormer import build_flowformer
        from third_parties.FlowFormer.core.utils.utils import (
            InputPadder as InputPadderFlowFormer,
        )
    else:
        raise ValueError

    model = build_model(args.model_to_use, args.ckpt_f, device)

    # unzip data
    root_dir = pathlib.Path(args.root_dir)

    if args.image_subdir != "null":
        raw_data_dir_list = list(root_dir.glob(args.image_subdir))
        assert (
            len(raw_data_dir_list) == 1
        ), f"{str(root_dir)}, {args.image_subdir}, {raw_data_dir_list}"
        raw_data_dir = raw_data_dir_list[0]
    else:
        raw_data_dir = root_dir

    save_dir = pathlib.Path(args.save_dir)
    save_dir.mkdir(exist_ok=True, parents=True)
    if args.for_colmap:
        data_dir = save_dir / "rgbs_for_colmap"
    else:
        data_dir = save_dir / "rgbs"

    data_dir.mkdir(exist_ok=True, parents=True)
    img_exts = PIL.Image.registered_extensions()
    supported_img_exts = {ex for ex, f in img_exts.items() if f in PIL.Image.OPEN}
    all_img_fs = []
    for tmp_ext in supported_img_exts:
        all_img_fs = all_img_fs + list(
            raw_data_dir.glob(f"{args.image_pattern}{tmp_ext}")
        )
    all_img_fs = sorted(all_img_fs)
    for tmp_f in tqdm.tqdm(all_img_fs, desc="copying RGBs"):
        if not (data_dir / tmp_f.name).exists():
            shutil.copyfile(tmp_f, data_dir / tmp_f.name)

    pose_f = root_dir / "poses_bounds_cvd.npy"
    if pose_f.exists():
        shutil.copyfile(pose_f, save_dir / pose_f.name)

    if args.for_colmap:
        out_dir = save_dir / "flows_for_colmap"
    else:
        out_dir = save_dir / "flows"
    out_dir.mkdir(exist_ok=True, parents=True)

    print("\n\ndata_dir: ", data_dir)
    print("out_dir: ", out_dir, data_dir, "\n\n")

    with torch.no_grad():
        run(
            device=device,
            input_dir=data_dir,
            out_dir=out_dir,
            model=model,
            sigma=0.05,
            img_pair_max_diff=args.img_pair_max_diff,
            max_hw=args.max_hw,
            model_to_use=args.model_to_use,
            flowformer_use_tile=args.flowformer_use_tile,
        )
