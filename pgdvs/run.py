#!/usr/bin/python3

import os
import random
import pathlib
import logging
import copy
import shutil
import numpy as np
from typing import Any

import hydra
import hydra.utils
from hydra import compose, initialize
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, read_write, open_dict

import torch

# The flag below controls whether to allow TF32 on matmul. This flag defaults to True.
torch.backends.cuda.matmul.allow_tf32 = False

# The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
torch.backends.cudnn.allow_tf32 = False

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"


LOGGER = logging.getLogger(__name__)
CONF_FP: str = str(pathlib.Path(__file__).absolute().parent.parent / "configs")


def _set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _find_free_port():
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def _save_config(cfg: DictConfig, filename: str, output_dir: pathlib.Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(str(output_dir / filename), "w", encoding="utf-8") as file:
        file.write(OmegaConf.to_yaml(cfg))


def _get_hydra_config_dir(hydra_config):
    if hydra_config.output_subdir is not None:
        hydra_config_dir = (
            pathlib.Path(hydra_config.runtime.output_dir) / hydra_config.output_subdir
        )
    else:
        hydra_config_dir = pathlib.Path(hydra_config.runtime.output_dir)
    return hydra_config_dir


def _manually_save_hydra_config(cfg, hydra_config, hydra_output_dir, all_cfg=None):
    # We mannually dump the configs
    # Ref: https://github.com/facebookresearch/hydra/blob/66c3b37bffde8a0b00258d032505eef7b5720fd4/hydra/core/utils.py#L175
    task_cfg = copy.deepcopy(cfg)
    with read_write(task_cfg):
        with open_dict(task_cfg):
            del task_cfg["hydra"]

    os.makedirs(hydra_output_dir, exist_ok=True)
    if all_cfg is not None:
        _save_config(all_cfg, "all.yaml", hydra_output_dir)
    _save_config(task_cfg, "config.yaml", hydra_output_dir)
    _save_config(hydra_config, "hydra.yaml", hydra_output_dir)
    _save_config(hydra_config.overrides.task, "overrides.yaml", hydra_output_dir)


def _distributed_worker(
    global_rank: int,
    world_size: int,
    cfg: DictConfig,
    hydra_config: DictConfig,
) -> float:
    if torch.cuda.is_available():
        # https://discuss.pytorch.org/t/extra-10gb-memory-on-gpu-0-in-ddp-tutorial/118113/2
        torch.cuda.set_device(global_rank)
        torch.cuda.empty_cache()
    else:
        assert not cfg.distributed

    hydra.core.utils.configure_log(hydra_config.job_logging, hydra_config.verbose)

    LOGGER.info("Distributed worker: %d / %d" % (global_rank + 1, world_size))

    hydra_config_dir = _get_hydra_config_dir(hydra_config)

    if global_rank == 0:
        LOGGER.info("cfg is:")
        LOGGER.info(OmegaConf.to_yaml(cfg))
        LOGGER.info(f"hydra_config: {OmegaConf.to_yaml(hydra_config)}")

        all_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
        _save_config(all_cfg, "all.yaml", hydra_config_dir)

    if cfg.percision == "float64":
        torch.set_default_tensor_type(torch.DoubleTensor)
        torch.set_default_dtype(torch.float64)

    if cfg.distributed:
        torch.cuda.set_device(global_rank)
        torch.distributed.init_process_group(
            backend="nccl", world_size=world_size, rank=global_rank
        )
    _set_random_seed(cfg.seed + global_rank)

    # run_dir = hydra_config.run.dir  # not working
    run_dir = hydra_config.runtime.output_dir
    hydra_log_f = hydra_config.job_logging.handlers.file.filename

    engine = hydra.utils.instantiate(
        cfg.engine,
        cfg,  # NOTE: cfg cannot be passed as key-word argument: this is the full config, it contains the _target_ of engine as well. Putting it into kwargs will cause error.
        hydra_config,
        global_rank=global_rank,
        local_rank=global_rank,
        world_size=world_size,
        run_dir=run_dir,
        hydra_config_dir=str(hydra_config_dir),
        hydra_log_f=hydra_log_f,
        flag_verbose=cfg.verbose,
    )

    if cfg.resume in ["eval", "eval_wo_resume"]:
        output = engine.run_eval()
    elif cfg.resume in ["none", "train", "vis", "vis_wo_resume"]:
        output = engine.run()
    else:
        raise ValueError(cfg.resume)

    if cfg.distributed:
        torch.distributed.barrier(device_ids=[global_rank])
        torch.distributed.destroy_process_group()

    LOGGER.info("Job Done for worker: %d / %d" % (global_rank + 1, world_size))
    return output


def run(cfg: DictConfig, hydra_config: DictConfig) -> float:
    assert cfg.percision in ["float32", "float64"]

    if cfg.distributed:
        assert torch.cuda.is_available()

        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(_find_free_port())
        world_size = torch.cuda.device_count()
        process_context = torch.multiprocessing.spawn(
            _distributed_worker,
            args=(
                world_size,
                cfg,
                hydra_config,
            ),
            nprocs=world_size,
            join=False,
        )
        try:
            process_context.join()
        except KeyboardInterrupt:
            # this is important.
            # if we do not explicitly terminate all launched subprocesses,
            # they would continue living even after this main process ends,
            # eventually making the OD machine unusable!
            for i, process in enumerate(process_context.processes):
                if process.is_alive():
                    LOGGER.info("terminating process " + str(i) + "...")
                    process.terminate()
                process.join()
                LOGGER.info("process " + str(i) + " finished")
        return 1.0
    else:
        return _distributed_worker(
            global_rank=0,
            world_size=1,
            cfg=cfg,
            hydra_config=hydra_config,
        )


@hydra.main(config_path=CONF_FP, config_name="pgdvs", version_base="1.2")
def cli(cfg: DictConfig):
    # cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))

    hydra_config = HydraConfig.get()

    if cfg.resume not in ["none", "eval_wo_resume", "vis_wo_resume"]:
        # For reproducing: see https://github.com/facebookresearch/hydra/issues/1805#issuecomment-1307567947
        assert cfg.resume_dir is not None

        resume_dir = pathlib.Path(cfg.resume_dir) / "hydra"
        if not resume_dir.exists():
            resume_dir = pathlib.Path(cfg.resume_dir) / ".hydra"
        assert resume_dir.exists(), resume_dir
        resume_hydra_config = OmegaConf.load(resume_dir / "hydra.yaml")
        LOGGER.info(f"Resume hydra configuration from {resume_dir / 'hydra.yaml'}")

        resume_config_name = resume_hydra_config.hydra.job.config_name

        resumed_overrides = OmegaConf.load(resume_dir / "overrides.yaml")  # a list
        LOGGER.info(
            f"Raw resumed hydra configuration overrides are {resumed_overrides}"
        )

        cur_hydra_config_dir = _get_hydra_config_dir(hydra_config)
        LOGGER.info(f"Current hydra configuration directory {cur_hydra_config_dir}")

        new_overrides = OmegaConf.load(cur_hydra_config_dir / "overrides.yaml")
        LOGGER.info(f"Newly-added hydra configuration overrides are {new_overrides}")

        # NOTE: we append new overrides to overwrite the previously one
        resumed_overrides.extend(new_overrides)

        from hydra.core.global_hydra import GlobalHydra

        GlobalHydra.instance().clear()

        # config_path: path relative to this file
        # config_dir: absolute path
        # with hydra.initialize(config_path=CONF_FP):
        with hydra.initialize_config_dir(version_base="1.2", config_dir=CONF_FP):
            cfg = hydra.compose(
                config_name=resume_config_name,
                overrides=resumed_overrides,
                return_hydra_config=True,
            )

            OmegaConf.resolve(cfg)  # NOTE: this is somehow a must-have

        LOGGER.info(f"[new] hydra_config: {OmegaConf.to_yaml(hydra_config)}")

        cfg.hydra.runtime.output_dir = hydra_config.runtime.output_dir
        hydra_config = cfg.hydra

        LOGGER.info(f"[resumed] hydra_config: {OmegaConf.to_yaml(hydra_config)}")

        shutil.copytree(
            cur_hydra_config_dir, cur_hydra_config_dir.parent / ".hydra_old"
        )
        _manually_save_hydra_config(
            cfg, hydra_config, cur_hydra_config_dir, all_cfg=None
        )

    run(cfg, hydra_config)


if __name__ == "__main__":
    cli()
