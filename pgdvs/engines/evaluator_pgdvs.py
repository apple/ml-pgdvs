import tqdm
import pickle
import logging
import pathlib
import PIL.Image
import numpy as np
from collections import defaultdict, OrderedDict

import torch

from pgdvs.engines.trainer_pgdvs import PGDVSTrainer
from pgdvs.utils.rendering import modify_rgb_range
from pgdvs.utils.training import calculate_psnr, calculate_ssim
from pgdvs.datasets.nvidia_eval import NvidiaDynEvaluationDataset
from pgdvs.datasets.dycheck_iphone_eval import DyCheckiPhoneEvaluationDataset


LOGGER = logging.getLogger(__name__)


class PGDVSEvaluator(PGDVSTrainer):
    def vis_model(
        self, epoch=None, total_steps=None, flag_full_vis=False, dataset_split="eval"
    ):
        pass

    @torch.no_grad()
    def eval_step(
        self,
        *,
        data,
        epoch,
        global_step,
        save_individual=True,
    ):
        data_gpu = self._to_gpu_func(data, self.device)

        self.model.eval()

        n_batch, n_context_temporal, raw_h, raw_w, _ = data["rgb_src_temporal"].shape

        ret_dict = self._get_model_module(self.model).forward(
            data_gpu,
            render_cfg=self.engine_cfg.render_cfg,
            disable_tqdm=not self.verbose,
            for_debug=False,
        )

        if False:
            self.debug_ret(ret_dict)

        rgb_pred_dict = OrderedDict(
            {"combined": ret_dict["combined_rgb"].clamp(0.0, 1.0)}
        )

        for k in rgb_pred_dict:
            if torch.any(torch.isnan(rgb_pred_dict[k])):
                tmp_scene_ids = [
                    data["misc"][_]["scene_id"] for _ in range(len(data["misc"]))
                ]
                tmp_tgt_frame_ids = [
                    data["misc"][_]["tgt_frame_id"] for _ in range(len(data["misc"]))
                ]
                LOGGER.info(
                    f"Found NaN for {k} of {tmp_scene_ids}, {tmp_tgt_frame_ids}\n"
                )

                rgb_pred_dict[k] = torch.nan_to_num(rgb_pred_dict[k], nan=0.0)

        rgb_gt = data_gpu["rgb_tgt"].permute(0, 3, 1, 2).clamp(0.0, 1.0)
        eval_mask = data_gpu["eval_mask"].permute(0, 3, 1, 2)

        # NOTE: we first quantize image then change it to float
        # to make sure the results are same with those computed by loading image from disk.
        rgb_gt = (rgb_gt * 255).byte().float() / 255.0
        for k in rgb_pred_dict:
            rgb_pred_dict[k] = (rgb_pred_dict[k] * 255).byte().float() / 255.0

        _, _, render_h, render_w = rgb_pred_dict["combined"].shape

        if (rgb_gt.shape[2] != render_h) or (rgb_gt.shape[3] != render_w):
            rgb_gt = torch.nn.functional.interpolate(
                rgb_gt,
                size=(render_h, render_w),
                mode="bicubic",
                antialias=True,
                align_corners=True,
            )
            eval_mask = torch.nn.functional.interpolate(
                eval_mask, size=(render_h, render_w), mode="nearest"
            )
            eval_mask = (eval_mask > 0).float()

        rgb_pred_dict_for_lpips = {
            k: modify_rgb_range(
                rgb_pred_dict[k],
                src_range=self.cfg.rgb_range,
                tgt_range="-1_1",
                check_range=False,
            )
            for k in rgb_pred_dict
        }  # [B, 3, H, W]

        rgb_gt_for_lpips = modify_rgb_range(
            rgb_gt,
            src_range=self.cfg.rgb_range,
            tgt_range="-1_1",
            check_range=False,
        )  # [B, 3, H, W]

        all_cnt = 0

        all_info_dict = defaultdict(list)

        # save individual image
        for i_b in tqdm.tqdm(np.arange(n_batch), disable=True):
            if "split" in data["misc"][i_b]:
                tmp_split = data["misc"][i_b]["split"]
            else:
                tmp_split = ""
            tmp_scene_id = data["misc"][i_b]["scene_id"]
            tmp_frame_id = data["misc"][i_b]["tgt_frame_id"]
            tmp_cam_id = data["misc"][i_b]["tgt_cam_id"]

            tmp_fname = f"{tmp_frame_id:05d}_cam_{tmp_cam_id:03d}"

            tmp_scene_info_dir = pathlib.Path(self.INFO_DIR) / tmp_split / tmp_scene_id
            tmp_scene_info_dir.mkdir(parents=True, exist_ok=True)
            tmp_info_f = tmp_scene_info_dir / f"{tmp_fname}_rank_{self.local_rank}.pkl"

            tmp_scene_vis_dir = pathlib.Path(self.VIS_DIR) / tmp_split / tmp_scene_id
            tmp_scene_vis_dir.mkdir(parents=True, exist_ok=True)
            tmp_vis_f = tmp_scene_vis_dir / f"{tmp_fname}_rank_{self.local_rank}.png"

            info_dict = {"src_frame_ids": data["seq_ids"][i_b, 1:].numpy()}

            if self.engine_cfg.quant_type == "nvidia":
                quant_func = self.obtain_quantitative_nvidia
            elif self.engine_cfg.quant_type == "dycheck_iphone":
                quant_func = self.obtain_quantitative_dycheck_iphone
            else:
                raise TypeError(type(self.datasets["eval"]))

            info_dict = quant_func(
                i_b=i_b,
                info_dict=info_dict,
                rgb_gt=rgb_gt,
                rgb_pred_dict=rgb_pred_dict,
                rgb_pred_dict_for_lpips=rgb_pred_dict_for_lpips,
                rgb_gt_for_lpips=rgb_gt_for_lpips,
                eval_mask=eval_mask,
                save_info_f=tmp_info_f,
                save_individual=save_individual,
            )

            if save_individual:
                self.save_vis_for_eval(
                    i_b=i_b,
                    data_gpu=data_gpu,
                    ret_dict=ret_dict,
                    rgb_pred_dict=rgb_pred_dict,
                    render_h=render_h,
                    render_w=render_w,
                    img_gt=rgb_gt[i_b : (i_b + 1), ...],
                    scene_vis_dir=tmp_scene_vis_dir,
                    save_fname=tmp_fname,
                    vis_f=tmp_vis_f,
                    scene_info_dir=tmp_scene_info_dir,
                )

            all_cnt += 1

            for info_k in info_dict:
                if info_k != "src_frame_ids":
                    all_info_dict[info_k].append(info_dict[info_k])

        metric_dict = {"eval/count": torch.LongTensor([all_cnt]).to(rgb_gt.device)}
        for info_k in all_info_dict:
            metric_dict[f"eval/{info_k}"] = (
                torch.FloatTensor(all_info_dict[info_k]).sum().to(rgb_gt.device)
            )

        for info_k in metric_dict:
            torch.distributed.reduce(
                metric_dict[info_k], dst=0, op=torch.distributed.ReduceOp.SUM
            )

        return metric_dict

    def obtain_quantitative_nvidia(
        self,
        *,
        i_b,
        info_dict,
        rgb_gt,
        rgb_pred_dict,
        rgb_pred_dict_for_lpips,
        rgb_gt_for_lpips,
        eval_mask,
        save_info_f,
        save_individual=True,
    ):
        for pred_k in rgb_pred_dict:
            rgb_pred = rgb_pred_dict[pred_k]
            rgb_pred_for_lpips = rgb_pred_dict_for_lpips[pred_k]

            tmp_pred = rgb_pred[i_b : (i_b + 1), ...]  # [1, 3, H, W]
            tmp_gt = rgb_gt[i_b : (i_b + 1), ...]

            # [H, W, 3], range [0, 1]
            tmp_pred_np = tmp_pred[0, ...].permute(1, 2, 0).cpu().numpy()
            tmp_gt_np = tmp_gt[0, ...].permute(1, 2, 0).cpu().numpy()

            # full
            tmp_full_mask = np.ones_like(
                tmp_gt_np, dtype=np.float32
            )  # NOTE: here must be [H, W, 3]
            tmp_psnr = calculate_psnr(tmp_gt_np, tmp_pred_np, tmp_full_mask)
            tmp_ssim = calculate_ssim(tmp_gt_np, tmp_pred_np, tmp_full_mask)

            tmp_lpips = self.lpips_fn.forward(
                rgb_gt_for_lpips[i_b : (i_b + 1), ...],
                rgb_pred_for_lpips[i_b : (i_b + 1), ...],
                torch.FloatTensor(tmp_full_mask)[None, ...]
                .permute(0, 3, 1, 2)
                .to(rgb_gt.device),
            ).item()

            # dynamic area
            tmp_eval_mask_dyn = (
                eval_mask[i_b, ...].permute(1, 2, 0).cpu().numpy()
            )  # NOTE: here must be [H, W, 3]
            tmp_psnr_dyn = calculate_psnr(tmp_gt_np, tmp_pred_np, tmp_eval_mask_dyn)
            tmp_ssim_dyn = calculate_ssim(tmp_gt_np, tmp_pred_np, tmp_eval_mask_dyn)

            tmp_lpips_dyn = self.lpips_fn.forward(
                rgb_gt_for_lpips[i_b : (i_b + 1), ...],
                rgb_pred_for_lpips[i_b : (i_b + 1), ...],
                eval_mask[i_b : (i_b + 1), ...],
            ).item()

            # static area
            tmp_eval_mask_static = (
                1.0 - tmp_eval_mask_dyn
            )  # NOTE: here must be [H, W, 3]
            tmp_psnr_static = calculate_psnr(
                tmp_gt_np, tmp_pred_np, tmp_eval_mask_static
            )
            tmp_ssim_static = calculate_ssim(
                tmp_gt_np, tmp_pred_np, tmp_eval_mask_static
            )

            tmp_lpips_static = self.lpips_fn.forward(
                rgb_gt_for_lpips[i_b : (i_b + 1), ...],
                rgb_pred_for_lpips[i_b : (i_b + 1), ...],
                1.0 - eval_mask[i_b : (i_b + 1), ...],
            ).item()

            info_dict.update(
                {
                    # full
                    f"psnr_full_{pred_k}": tmp_psnr,
                    f"ssim_full_{pred_k}": tmp_ssim,
                    f"lpips_full_{pred_k}": tmp_lpips,
                    # dyn
                    f"psnr_dyn_{pred_k}": tmp_psnr_dyn,
                    f"ssim_dyn_{pred_k}": tmp_ssim_dyn,
                    f"lpips_dyn_{pred_k}": tmp_lpips_dyn,
                    # static
                    f"psnr_static_{pred_k}": tmp_psnr_static,
                    f"ssim_static_{pred_k}": tmp_ssim_static,
                    f"lpips_static_{pred_k}": tmp_lpips_static,
                }
            )

        if save_individual:
            with open(save_info_f, "wb") as f:
                pickle.dump(info_dict, f)

        return info_dict

    def obtain_quantitative_dycheck_iphone(
        self,
        *,
        i_b,
        info_dict,
        rgb_gt,
        rgb_pred_dict,
        rgb_pred_dict_for_lpips,
        rgb_gt_for_lpips,
        eval_mask,
        save_info_f,
        save_individual=True,
    ):
        # lazy import as jax may not be installed outside `dycheck_iphone` quant_type
        import pgdvs.utils.dycheck.metrics as dycheck_metrics
        import jax

        device_cpu = jax.devices("cpu")[0]

        for pred_k in rgb_pred_dict:
            rgb_pred = rgb_pred_dict[pred_k]

            # rgb_pred_for_lpips = rgb_pred_dict_for_lpips[pred_k]

            tmp_pred = rgb_pred[i_b : (i_b + 1), ...]  # [1, 3, H, W]
            tmp_gt = rgb_gt[i_b : (i_b + 1), ...]

            # [H, W, 3], range [0, 1]
            tmp_pred_np = tmp_pred[0, ...].permute(1, 2, 0).cpu().numpy()
            tmp_gt_np = tmp_gt[0, ...].permute(1, 2, 0).cpu().numpy()

            tmp_full_mask = np.ones_like(tmp_gt_np, dtype=np.float32)[
                ..., :1
            ]  # NOTE: here must be [H, W, 1]

            # pred / gt: [H, W, 3], float32, range [0, 1]
            # covis_mask: [H, W, 3], float32, range [0, 1]

            with jax.default_device(device_cpu):
                tmp_psnr = dycheck_metrics.compute_psnr(
                    tmp_gt_np, tmp_pred_np, tmp_full_mask, local_rank=self.local_rank
                ).item()
                tmp_ssim = dycheck_metrics.compute_ssim(
                    tmp_gt_np, tmp_pred_np, tmp_full_mask, local_rank=self.local_rank
                ).item()
                tmp_lpips = dycheck_metrics.compute_lpips(
                    self.lpips_fn,
                    tmp_gt_np,
                    tmp_pred_np,
                    tmp_full_mask,
                    # device=torch.device("cpu"),
                    device=rgb_pred.device,
                ).item()

            # with covis mask
            tmp_eval_mask_np = eval_mask[i_b, ...].permute(1, 2, 0).cpu().numpy()

            tmp_mpsnr = dycheck_metrics.compute_psnr(
                tmp_gt_np, tmp_pred_np, tmp_eval_mask_np, local_rank=self.local_rank
            ).item()
            tmp_mssim = dycheck_metrics.compute_ssim(
                tmp_gt_np, tmp_pred_np, tmp_eval_mask_np, local_rank=self.local_rank
            ).item()
            tmp_mlpips = dycheck_metrics.compute_lpips(
                self.lpips_fn,
                tmp_gt_np,
                tmp_pred_np,
                tmp_eval_mask_np,
                device=rgb_pred.device,
            ).item()

            # full
            # tmp_full_mask = np.ones_like(
            #     tmp_gt_np, dtype=np.float32
            # )  # NOTE: here must be [H, W, 3]
            # tmp_psnr = calculate_psnr(tmp_gt_np, tmp_pred_np, tmp_full_mask)
            # tmp_ssim = calculate_ssim(tmp_gt_np, tmp_pred_np, tmp_full_mask)

            # tmp_lpips = self.lpips_fn.forward(
            #     rgb_gt_for_lpips[i_b : (i_b + 1), ...],
            #     rgb_pred_for_lpips[i_b : (i_b + 1), ...],
            #     torch.FloatTensor(tmp_full_mask)[None, ...]
            #     .permute(0, 3, 1, 2)
            #     .to(rgb_gt.device),
            # ).item()

            # # dynamic area
            # tmp_eval_mask_dyn = (
            #     eval_mask[i_b, ...].permute(1, 2, 0).cpu().numpy()
            # )  # NOTE: here must be [H, W, 3]
            # tmp_psnr_dyn = calculate_psnr(tmp_gt_np, tmp_pred_np, tmp_eval_mask_dyn)
            # tmp_ssim_dyn = calculate_ssim(tmp_gt_np, tmp_pred_np, tmp_eval_mask_dyn)

            # tmp_lpips_dyn = self.lpips_fn.forward(
            #     rgb_gt_for_lpips[i_b : (i_b + 1), ...],
            #     rgb_pred_for_lpips[i_b : (i_b + 1), ...],
            #     eval_mask[i_b : (i_b + 1), ...],
            # ).item()

            # # static area
            # tmp_eval_mask_static = (
            #     1.0 - tmp_eval_mask_dyn
            # )  # NOTE: here must be [H, W, 3]
            # tmp_psnr_static = calculate_psnr(
            #     tmp_gt_np, tmp_pred_np, tmp_eval_mask_static
            # )
            # tmp_ssim_static = calculate_ssim(
            #     tmp_gt_np, tmp_pred_np, tmp_eval_mask_static
            # )

            # tmp_lpips_static = self.lpips_fn.forward(
            #     rgb_gt_for_lpips[i_b : (i_b + 1), ...],
            #     rgb_pred_for_lpips[i_b : (i_b + 1), ...],
            #     1.0 - eval_mask[i_b : (i_b + 1), ...],
            # ).item()

            info_dict.update(
                {
                    # full
                    f"psnr_{pred_k}": tmp_psnr,
                    f"ssim_{pred_k}": tmp_ssim,
                    f"lpips_{pred_k}": tmp_lpips,
                    # covis
                    f"mpsnr_{pred_k}": tmp_mpsnr,
                    f"mssim_{pred_k}": tmp_mssim,
                    f"mlpips_{pred_k}": tmp_mlpips,
                }
            )

        if save_individual:
            with open(save_info_f, "wb") as f:
                pickle.dump(info_dict, f)

        return info_dict

    def save_vis_for_eval(
        self,
        *,
        i_b,
        data_gpu,
        ret_dict,
        rgb_pred_dict,
        render_h,
        render_w,
        img_gt,
        scene_vis_dir,
        save_fname,
        vis_f,
        scene_info_dir,
    ):
        img_gt = (img_gt[0, ...].permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
        PIL.Image.fromarray(img_gt).save(scene_vis_dir / f"{save_fname}_gt.png")

        img_combined = (
            rgb_pred_dict["combined"][i_b, ...].permute(1, 2, 0).cpu().numpy() * 255
        ).astype(np.uint8)
        PIL.Image.fromarray(img_combined).save(
            scene_vis_dir / f"{save_fname}_combined.png"
        )

        if "static_coarse_rgb" in ret_dict:
            # save pure GNT results
            img_gnt = (
                ret_dict["static_coarse_rgb"][i_b, ...]
                .clamp(0.0, 1.0)
                .permute(1, 2, 0)
                .cpu()
                .numpy()
                * 255
            ).astype(np.uint8)
            PIL.Image.fromarray(img_gnt).save(scene_vis_dir / f"{save_fname}_gnt.png")
        if "geo_static_rgb" in ret_dict:
            # save pure geometric results
            img_gnt = (
                ret_dict["geo_static_rgb"][i_b, ...]
                .clamp(0.0, 1.0)
                .permute(1, 2, 0)
                .cpu()
                .numpy()
                * 255
            ).astype(np.uint8)
            PIL.Image.fromarray(img_gnt).save(
                scene_vis_dir / f"{save_fname}_geo_static.png"
            )
