import os
import socket
import random
import boto3
import botocore
import logging
import numpy as np
from abc import abstractmethod
from omegaconf import DictConfig

import torch

from pgdvs.utils.training import resume_from_ckpt, upload_to_s3

LOGGER = logging.getLogger(__name__)


def combine(l):
    if isinstance(l[0], torch.Tensor):
        return torch.stack(l, dim=0)
    if isinstance(l[0], float):
        return torch.Tensor(l)
    if isinstance(l[0], list) and isinstance(l[0][0], float):
        return torch.Tensor(l)
    else:
        return l


def default_collate_fn(batch):
    return {k: combine([x[k] for x in batch]) for k in batch[0].keys()}


class AbstractEngine:
    def __init__(
        self,
        cfg: DictConfig,
        hydra_config: DictConfig,
        *,
        global_rank: int,
        local_rank: int,
        world_size: int,
        run_dir: str,
        hydra_config_dir: str = None,
        hydra_log_f: str = None,
        is_training: bool = False,
        flag_verbose: bool = True,
    ) -> None:
        self.global_rank = global_rank
        self.local_rank = local_rank
        self.world_size = world_size

        self.cfg = cfg
        self.hydra_config = hydra_config

        if torch.cuda.is_available():
            self.device = "cuda:%d" % local_rank
        else:
            self.device = torch.device("cpu")

        self.is_training = is_training

        self.verbose = flag_verbose

        self.hydra_log_f = hydra_log_f

        self.LOG_DIR = run_dir  # Essentially hydra_config.runtime.output_dir

        self.INFO_DIR = os.path.join(self.LOG_DIR, f"infos")
        self.CHECKPOINT_FOLDER = os.path.join(self.LOG_DIR, "checkpoints")
        self.TENSORBOARD_DIR = os.path.join(self.LOG_DIR, "tb")
        self.VIS_DIR = os.path.join(self.LOG_DIR, "vis")

        os.makedirs(self.INFO_DIR, exist_ok=True)
        os.makedirs(self.CHECKPOINT_FOLDER, exist_ok=True)
        os.makedirs(self.TENSORBOARD_DIR, exist_ok=True)
        os.makedirs(self.VIS_DIR, exist_ok=True)

        LOGGER.info("All files are saved to %s" % self.LOG_DIR)

        # setup model
        self.model_modules_not_to_save = None  # this must be placed before build_model
        self.model, self.optimizer = self.build_model()

        self.init_epoch = 0
        self.init_total_steps = 0
        self.init_total_steps_on_epoch_start = 0

        self.max_epochs = self.cfg.max_epochs

        self.s3_info = None

        if (
            self.cfg.resume
            not in ["none", "elastic", "eval_wo_resume", "vis_wo_resume"]
        ) or (self.cfg.resume in ["eval"] and not self.cfg.series_eval):
            self.run_resume(self.cfg.resume_epoch)

        if isinstance(self.model, dict):
            for k in self.model:
                self.model[k].to(self.device)
        else:
            self.model.to(self.device)

        # setup dataset
        self.datasets = self.build_dataset()
        LOGGER.info(
            f"Initialization done in Rank {self.global_rank} | Local rank {self.local_rank}"
        )

    def run_resume(self, resume_epoch):
        if self.cfg.resume_dir is not None:
            # load other pretrain models.
            ckpt_dir = self.cfg.resume_dir
        else:
            # resume from the default path.
            ckpt_dir = self.CHECKPOINT_FOLDER

        (
            self.init_epoch,
            _,
            self.init_total_steps,
            self.init_total_steps_on_epoch_start,
        ) = resume_from_ckpt(
            ckpt_dir=ckpt_dir,
            model=self.model,
            modules_not_saved=self.model_modules_not_to_save,
            optimizer=self.optimizer if self.cfg.resume_dir is not None else None,
            epoch=resume_epoch,
            strict=False,
            cfg=self.cfg,
            device=self.device,
        )

    def upload_info_to_s3(self):
        for cur_root, cur_dirs, cur_files in os.walk(self.LOG_DIR):
            for tmp_f in cur_files:
                local_f = os.path.join(cur_root, tmp_f)
                if ("tb" in local_f) or ("hydra" in local_f):
                    upload_to_s3(local_f, self.LOG_DIR, **self.s3_info)

    def run_preparation(self):
        pass

    def _set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

    def _to_gpu_func(self, batch, device):
        return {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

    @abstractmethod
    def build_model(self):
        raise NotImplementedError

    @abstractmethod
    def build_dataset(self):
        raise NotImplementedError
