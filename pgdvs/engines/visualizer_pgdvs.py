import sys
import logging
import pathlib
import tqdm
import traceback
import imageio_ffmpeg
import PIL.Image
import numpy as np
from collections import OrderedDict
import matplotlib.pyplot as plt

import torch
import torchvision

from pgdvs.engines.abstract import default_collate_fn
from pgdvs.engines.trainer_pgdvs import PGDVSTrainer
from pgdvs.utils.rendering import images_to_video


LOGGER = logging.getLogger(__name__)


class PGDVSVisualizer(PGDVSTrainer):
    @torch.no_grad()
    def run(self):
        self.vis_model()

    @torch.no_grad()
    def vis_model(self, epoch=None, flag_full_vis=False, dataset_split="vis"):
        try:
            LOGGER.info("Check FFMPEG exists.")
            imageio_ffmpeg.get_ffmpeg_exe()
            flag_ffmpeg_exe = True
        except RuntimeError:
            traceback.print_exc()
            err = sys.exc_info()[0]
            LOGGER.info(err)
            LOGGER.info(
                f"FFMPEG is not properly set therefore we do not automatically generate videos."
            )
            flag_ffmpeg_exe = False

        sampler = torch.utils.data.distributed.DistributedSampler(
            self.datasets[dataset_split],
            num_replicas=self.world_size,
            rank=self.global_rank,
            shuffle=False,
        )

        batch_size_per_proc = self.cfg.eval_batch_size // self.world_size
        real_eval_batch_size = batch_size_per_proc * self.world_size

        eval_dataloader = torch.utils.data.DataLoader(
            self.datasets[dataset_split],
            sampler=sampler,
            num_workers=self.cfg.n_dataloader_workers,
            batch_size=batch_size_per_proc,  # self.cfg.eval_batch_size,
            collate_fn=default_collate_fn,
        )

        if self.cfg.n_max_eval_data > 0:
            n_all_data = min(
                len(self.datasets[dataset_split]), self.cfg.n_max_eval_data
            )
        else:
            n_all_data = len(self.datasets[dataset_split])
        n_batches_per_epoch = int(np.ceil(n_all_data / real_eval_batch_size))

        LOGGER.info(
            f"Rank {self.global_rank} | Local rank {self.local_rank} batch: #per_epoch {n_batches_per_epoch}, "
            f"#all_data {n_all_data}, #batch_size_per_proc {batch_size_per_proc}, #real_batch_size {real_eval_batch_size}"
        )

        vis_dir_dict = {}

        for step, data in tqdm.tqdm(
            enumerate(eval_dataloader),
            total=n_batches_per_epoch,
            disable=not self.is_main_proc,
            leave=False,
        ):
            if step >= n_batches_per_epoch:
                break

            data_gpu = self._to_gpu_func(data, self.device)

            self.model.eval()

            n_batch, n_context, raw_h, raw_w, _ = data["rgb_src_temporal"].shape

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

            for i_b in tqdm.tqdm(np.arange(n_batch), disable=True):
                if "split" in data["misc"][i_b]:
                    tmp_split = data["misc"][i_b]["split"]
                else:
                    tmp_split = ""
                tmp_scene_id = data["misc"][i_b]["scene_id"]
                tmp_tgt_idx = data["misc"][i_b]["tgt_idx"]

                tmp_fname = f"{tmp_tgt_idx:05d}"

                tmp_scene_vis_dir = (
                    pathlib.Path(self.VIS_DIR) / tmp_split / tmp_scene_id
                )
                tmp_scene_vis_dir.mkdir(parents=True, exist_ok=True)

                vis_dir_dict[tmp_scene_id] = tmp_scene_vis_dir

                tmp_vis_combined_f = tmp_scene_vis_dir / f"{tmp_fname}_combined.png"
                torchvision.utils.save_image(
                    rgb_pred_dict["combined"][i_b : (i_b + 1), ...], tmp_vis_combined_f
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
                    PIL.Image.fromarray(img_gnt).save(
                        tmp_scene_vis_dir / f"{tmp_fname}_gnt.png"
                    )

        if flag_ffmpeg_exe:
            try:
                LOGGER.info("Converting images to video.")

                for tmp_scene_id in vis_dir_dict:
                    tmp_vis_dir = vis_dir_dict[tmp_scene_id]

                    tmp_combined_save_f = (
                        tmp_scene_vis_dir.parent / f"{tmp_scene_id}_combined.mp4"
                    )
                    self._compute_video(
                        tmp_vis_dir, "*_combined.png", tmp_combined_save_f
                    )
            except:
                traceback.print_exc()
                err = sys.exc_info()[0]
                LOGGER.info(err)
                LOGGER.info(f"Video generation fails.")

    def _compute_video(self, data_dir, glob_str, save_f, disable_tqdm=False):
        raw_combined_vis_f_list = sorted(
            list(data_dir.glob(glob_str))
        )  # e.g., combined_00062_rank_6.png
        tmp_dict = {}
        for tmp_f in raw_combined_vis_f_list:
            tmp_idx = int(tmp_f.stem.split("_")[1])
            tmp_dict[tmp_idx] = np.array(PIL.Image.open(tmp_f))

        tmp_all_imgs = [tmp_dict[_] for _ in sorted(list(tmp_dict.keys()))]
        images_to_video(
            tmp_all_imgs,
            save_f.parent,
            save_f.stem,
            fps=10,
            quality=9,
            disable_tqdm=disable_tqdm,
        )
