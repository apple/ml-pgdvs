import os
import tqdm
import logging
import pathlib
import hydra.utils
import PIL.Image
import numpy as np
from omegaconf import DictConfig
from collections import deque, defaultdict

import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel as DDP

from pgdvs.utils.vis_utils import colorize
from pgdvs.engines.abstract import AbstractEngine, default_collate_fn


# We need this otherwise somehow some dataloader works will be aborted
#   due to our vis_model operation in the main process.
#   That process will take some time so that the checker may think the worker is dead.
# Ref: https://github.com/pytorch/pytorch/issues/43455#issuecomment-678721860
torch.utils.data._utils.MP_STATUS_CHECK_INTERVAL = 600


TINY_NUMBER = 1e-6


LOGGER = logging.getLogger(__name__)


class PGDVSTrainer(AbstractEngine):
    def __init__(
        self,
        cfg: DictConfig,
        hydra_config: DictConfig,
        *,
        engine_cfg: None,
        global_rank: int = 0,
        local_rank: int = 0,
        world_size: int = 1,
        run_dir: str = None,
        hydra_config_dir: str = None,
        hydra_log_f: str = None,
        flag_verbose: bool = True,
    ) -> None:
        if torch.distributed.is_initialized():
            self.is_main_proc = torch.distributed.get_rank() == 0
        else:
            self.is_main_proc = True

        self.engine_cfg = engine_cfg

        super().__init__(
            cfg,
            hydra_config,
            global_rank=global_rank,
            local_rank=local_rank,
            world_size=world_size,
            run_dir=run_dir,
            hydra_config_dir=hydra_config_dir,
            hydra_log_f=hydra_log_f,
            is_training=True,
            flag_verbose=flag_verbose,
        )

        self.use_tb_local = True

        # setup tensorboard
        # This must be placed after the model resuming since we need `self.init_step`` to be updated.
        self.tb_dir = os.path.join(
            self.TENSORBOARD_DIR, f"global_rank_{self.global_rank}"
        )

        if self.is_main_proc:
            self.tb_writer = SummaryWriter(
                log_dir=self.tb_dir,
                purge_step=self.init_total_steps,
            )

            self.tb_writer.add_text("cfg", str(self.cfg), 0)
        else:
            self.tb_writer = None

        if self.engine_cfg.quant_type == "nvidia":
            import pgdvs.utils.nsff_lpips as lpips
        elif self.engine_cfg.quant_type == "dycheck_iphone":
            import lpips
        else:
            raise ValueError(self.engine_cfg.quant_type)

        if self.is_main_proc and (self.engine_cfg.quant_type == "dycheck_iphone"):
            # NOTE: we call LPIPS here to let it download pretrained checkpoints from the main process
            # to avoid several processes simultaneously download.
            _ = lpips.LPIPS(net="alex")

        if self.cfg.distributed:
            self.set_params_to_be_ignored_ddp(self.model)

            # https://github.com/pytorch/pytorch/blob/3179c21286e185f2e22d504d783b515dfaf2a600/torch/nn/parallel/distributed.py#L660-L682
            n_param_grads = 0
            if hasattr(self.model, "_ddp_params_and_buffers_to_ignore"):
                tmp_params_to_ignore = self.model._ddp_params_and_buffers_to_ignore
            else:
                tmp_params_to_ignore = []
            for name, param in self.model.named_parameters():
                if (name not in tmp_params_to_ignore) and param.requires_grad:
                    n_param_grads += 1

            if n_param_grads > 0:
                if isinstance(self.model, torch.nn.Module):
                    self.model = DDP(self.model, device_ids=[self.local_rank])
                elif isinstance(self.model, dict):
                    model_k_list = list(self.model.keys())
                    for k in model_k_list:
                        # Ref:
                        # - https://github.com/NVlabs/FUNIT/issues/23#issuecomment-997770973
                        # - https://github.com/NVlabs/stylegan2-ada-pytorch/blob/6f160b3d22b8b178ebe533a50d4d5e63aedba21d/training/training_loop.py#L187
                        self.model[k] = DDP(
                            self.model[k],
                            device_ids=[self.local_rank],
                            broadcast_buffers=False,
                        )
                else:
                    raise TypeError(type(self.model))

                torch.distributed.barrier(device_ids=[self.local_rank])  # sync

        if self.engine_cfg.quant_type == "nvidia":
            # https://github.com/google/dynibar/blob/02b164144cce2d93aa4c5d87b418497286b2ae31/eval_nvidia.py#L289-L291
            self.lpips_fn = lpips.PerceptualLoss(
                model="net-lin",
                net="alex",
                use_gpu=False,
                version=0.1,
            ).to(self.device)
        elif self.engine_cfg.quant_type == "dycheck_iphone":
            self.lpips_fn = lpips.LPIPS(net="alex", spatial=True).to(self.device)
        else:
            raise ValueError(self.engine_cfg.quant_type)

    def build_model(self):
        LOGGER.info("Creating Model.")
        current_seed = torch.initial_seed()

        # NOTE: we make sure the model has same parameters for multi-gpus.
        # Though this may be unnecessary since DDP will broadcast the weights from rank0 to all processes.
        # - [pytorch 1.10] https://github.com/pytorch/pytorch/blob/71f889c7d265b9636b93ede9d651c0a9c4bee191/torch/nn/parallel/distributed.py#L580
        # - [pytorch 1.2] https://github.com/pytorch/pytorch/blob/81e70ffa19c90e14c20d1369aae86d19f305dc7d/torch/nn/parallel/distributed.py#L299
        torch.manual_seed(self.cfg.seed)
        model = hydra.utils.instantiate(
            self.cfg.model,
            self.cfg,
            train_static_renderer=False,
            render_cfg=self.engine_cfg.render_cfg,
            local_rank=self.local_rank,
        ).to(self.device)

        self.model_modules_not_to_save = ["static_renderer", "dyn_renderer"]

        if self.is_main_proc:
            LOGGER.info(f"Model is {str(model)}\n")

            total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            LOGGER.info(f"Model's #param is {total_params}\n")

        optimizer = None

        # NOTE: resume the seed
        torch.manual_seed(current_seed)

        return model, optimizer

    def set_params_to_be_ignored_ddp(self, model):
        assert isinstance(model, torch.nn.Module), f"{type(model)}"

        params_to_ignore = []

        # remove some parameters from param_list.
        # - https://github.com/pytorch/pytorch/blob/8ca7820e4531e61b3d381d5eddf43c4969ba0c7d/torch/nn/parallel/distributed.py#L1701
        # - https://github.com/pytorch/pytorch/blob/181afd52205d974ff99d30dfa3d3033a71a47e2e/torch/testing/_internal/distributed/distributed_test.py#L3387-L3395
        for tmp_module in [model.static_renderer, model.dyn_renderer]:
            if tmp_module is not None:
                tmp_module_name = [
                    module_name
                    for module_name, module in model.named_modules()
                    if module is tmp_module
                ][0]
                params_to_ignore = params_to_ignore + [
                    f"{tmp_module_name}.{param_name}"
                    for param_name, _ in tmp_module.named_parameters()
                ]

        if self.is_main_proc:
            LOGGER.info(f"\nparams_to_ignore: {params_to_ignore}\n")

        DDP._set_params_and_buffers_to_ignore_for_model(
            model,
            params_to_ignore,
        )

    def build_dataset(self):
        self._set_seed(self.cfg.seed + self.global_rank)

        LOGGER.info("Creating Dataset.")

        all_datasets = {}

        if self.cfg.resume in ["none", "train"]:
            all_datasets["train"] = hydra.utils.instantiate(
                self.cfg.dataset, mode="train"
            )

        if self.cfg.resume in ["vis", "vis_wo_resume"]:
            all_datasets["vis"] = hydra.utils.instantiate(self.cfg.dataset, mode="vis")

        if self.cfg.resume in ["eval", "eval_wo_resume"]:
            all_datasets["eval"] = hydra.utils.instantiate(
                self.cfg.dataset, mode="eval"
            )

        return all_datasets

    def _get_model_module(self, model):
        if isinstance(model, DDP):
            tmp_module = model.module  # due to DDP
        elif isinstance(model, torch.nn.Module):
            tmp_module = model
        else:
            raise TypeError(type(model))
        return tmp_module

    def vis_model(self, epoch=None, flag_full_vis=True):
        raise NotImplementedError

    def run(self) -> float:
        if self.engine_cfg.for_overfit:
            self.run_overfit()
        else:
            self.run_train()

    def run_overfit(self):
        raise NotImplementedError

    def run_train(self) -> float:
        raise NotImplementedError

    def _write_log(self, name, val, step):
        assert self.is_main_proc
        if isinstance(val, torch.Tensor):
            val = val.item()
        self.tb_writer.add_scalar(name, val, step)

    def run_eval(self):
        if self.cfg.series_eval:
            ckpt_paths = list(
                pathlib.Path(self.cfg.resume_dir).glob(f"epoch_*.pth")
            ) + list(
                pathlib.Path(self.cfg.resume_dir).glob(f"checkpoints/epoch_*.pth")
            )  # e.g., epoch_000401-step_000023659.pth
            resume_epochs = sorted(
                [int(_.stem.split("epoch_")[1].split("-")[0]) for _ in ckpt_paths]
            )
        else:
            resume_epochs = [self.cfg.resume_epoch]

        LOGGER.info(f"Resume_epochs: {resume_epochs}")

        for tmp_epoch in tqdm.tqdm(resume_epochs):
            if self.cfg.resume not in ["eval_wo_resume", "vis_wo_resume"]:
                self.run_resume(tmp_epoch)
            if self.cfg.distributed:
                torch.distributed.barrier(device_ids=[self.local_rank])

            self.run_eval_single_ckpt(
                self.init_epoch,
                self.init_total_steps,
                save_individual=self.cfg.eval_save_individual,
            )

    @torch.no_grad()
    def run_eval_single_ckpt(
        self, epoch, total_steps, during_train=False, save_individual=False
    ):
        LOGGER.info(
            f"Evaluating model in Rank {self.global_rank} | Local rank {self.local_rank}."
        )

        sampler = torch.utils.data.distributed.DistributedSampler(
            self.datasets["eval"],
            num_replicas=self.world_size,
            rank=self.global_rank,
            shuffle=False,
        )

        batch_size_per_proc = self.cfg.eval_batch_size // self.world_size
        real_eval_batch_size = batch_size_per_proc * self.world_size

        eval_dataloader = torch.utils.data.DataLoader(
            self.datasets["eval"],
            sampler=sampler,
            num_workers=self.cfg.n_dataloader_workers,
            batch_size=batch_size_per_proc,  # self.cfg.eval_batch_size,
            collate_fn=default_collate_fn,
        )

        stats_trace = deque([], maxlen=100)

        if self.cfg.n_max_eval_data > 0:
            n_all_data = min(len(self.datasets["eval"]), self.cfg.n_max_eval_data)
        else:
            n_all_data = len(self.datasets["eval"])
        n_batches_per_epoch = int(np.ceil(n_all_data / real_eval_batch_size))

        LOGGER.info(
            f"Rank {self.global_rank} | Local rank {self.local_rank} batch: #per_epoch {n_batches_per_epoch}, "
            f"#all_data {n_all_data}, #batch_size_per_proc {batch_size_per_proc}, #real_batch_size {real_eval_batch_size}"
        )

        loss_sum = defaultdict(float)

        for step, data in tqdm.tqdm(
            enumerate(eval_dataloader),
            total=n_batches_per_epoch,
            disable=not self.is_main_proc,
            leave=False,
        ):

            if step >= n_batches_per_epoch:
                break

            stats = self.eval_step(
                data=data,
                epoch=epoch,
                global_step=total_steps,
                save_individual=save_individual,
            )
            stats_trace.append(stats)

            for tmp_k in stats:
                if "eval" in tmp_k:
                    loss_sum[tmp_k] = loss_sum[tmp_k] + stats[tmp_k].cpu()

        if self.is_main_proc:
            for tmp_k in loss_sum:
                tmp_avg = loss_sum[tmp_k] / loss_sum["eval/count"]
                self._write_log(tmp_k, tmp_avg, total_steps)
                LOGGER.info(f"[Epoch {epoch}] {tmp_k}: {tmp_avg}\n")

        self.vis_model(
            epoch=epoch,
            total_steps=total_steps,
            flag_full_vis=True,
            dataset_split="eval",
        )

        LOGGER.info(
            f"Finished evaluating in Rank {self.global_rank} | Local rank {self.local_rank}!"
        )

    @torch.no_grad()
    def eval_step(
        self,
        *,
        data,
        epoch,
        global_step,
        save_individual=False,
    ):
        raise NotImplementedError

    def save_img(self, img, save_fname, save_dir=None, nrow=4):
        img = torchvision.utils.make_grid(
            img,
            nrow=nrow,
            value_range=[0, 1],
            padding=0,
            pad_value=1.0,
        )
        if save_dir is None:
            save_dir = pathlib.Path(self.VIS_DIR)
        torchvision.utils.save_image(img, save_dir / save_fname)

    def debug_ret(self, ret_dict):
        for k in [
            "static_coarse_rgb",
            "static_coarse_inbound_cnt",
            "static_coarse_oob_mask",
            "static_coarse_dyn_cnt",
            "static_coarse_dyn_mask_any",
            "static_coarse_dyn_mask_all",
            "static_coarse_dyn_mask_thres",
            "geo_static_rgb",
            "geo_static_mask",
        ]:
            if k in ret_dict:
                self.save_img(ret_dict[k], f"1_{k}.png")

        for k in [
            "static_coarse_view_entropy",
            "static_coarse_view_std",
            "static_coarse_view_std_normalized",
        ]:
            if k in ret_dict:
                self.debug_vis_entropy_or_std(ret_dict[k][0, ...], f"1_{k}")

        if "static_fine_rgb" in ret_dict:
            for k in [
                "static_fine_rgb",
                "static_fine_inbound_cnt",
                "static_fine_oob_mask",
                "static_fine_dyn_cnt",
                "static_fine_dyn_mask_any",
                "static_fine_dyn_mask_all",
                "static_fine_dyn_mask_thres",
            ]:
                if k in ret_dict:
                    self.save_img(ret_dict[k], f"1_{k}.png")

            for k in [
                "static_fine_view_entropy",
                "static_fine_view_std",
                "static_fine_view_std_normalized",
            ]:
                if k in ret_dict:
                    self.debug_vis_entropy_or_std(ret_dict[k][0, ...], f"1_{k}")

        for k in [
            "render_dyn_rgb",
            "render_dyn_mask",
            "render_dyn_temporal_closest_rgb",
            "render_dyn_temporal_closest_mask",
            "render_dyn_temporal_track_rgb",
            "render_dyn_temporal_track_mask",
            "render_dyn_mask_holes_filled",
            "render_dyn_mask_to_be_inpaint",
        ]:
            if k in ret_dict:
                self.save_img(ret_dict[k], f"2_{k}.png")

        for k in [
            "combined_rgb",
            "combined_rgb_static",
            "combined_rgb_dyn",
        ]:
            if k in ret_dict:
                self.save_img(ret_dict[k], f"3_{k}.png")

        import sys

        sys.exit(1)

    def debug_vis_entropy_or_std(
        self,
        in_tensor,
        name,
        save_dir=None,
        grid_nrow=3,
        grid_pad_value=1.0,
    ):
        if save_dir is None:
            save_dir = pathlib.Path(self.VIS_DIR)

        in_tensor = in_tensor[:, None, ...]  # [#layer, H, W] -> [#layer, 1, H, W]
        min_val = torch.min(in_tensor).item()
        max_val = torch.max(in_tensor).item()
        # normalized_in_tensor = (in_tensor - torch.min(in_tensor)) / (torch.max(in_tensor) - torch.min(in_tensor) + 1e-8)
        normalized_in_tensor = (in_tensor - min_val) / (max_val - min_val + TINY_NUMBER)
        assert torch.all(
            (normalized_in_tensor <= 1.0) & (normalized_in_tensor >= 0)
        ), f"{torch.min(normalized_in_tensor)}, {torch.max(normalized_in_tensor)}"
        grid_img_in_tensor = torchvision.utils.make_grid(
            normalized_in_tensor,
            nrow=grid_nrow,
            value_range=[0, 1],
            padding=10,
            pad_value=grid_pad_value,
        )  # [3, H, W]
        torchvision.utils.save_image(
            grid_img_in_tensor,
            save_dir / f"{name}.png",
        )

        grid_img_in_tensor = grid_img_in_tensor.cpu().permute(1, 2, 0)[..., 0]  # [H, W]
        grid_img_in_tensor_cbar = colorize(
            grid_img_in_tensor,
            cmap_name="jet",
            append_cbar=True,
            range=[min_val, max_val],
        )
        PIL.Image.fromarray(
            (grid_img_in_tensor_cbar.cpu().numpy() * 255).astype(np.uint8)
        ).save(save_dir / f"{name}_cbar.png")
