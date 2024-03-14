import os
import cv2
import math
import pathlib
import logging
import skimage
import numpy as np

import torch
from torch.nn.parallel import DistributedDataParallel as DDP


LOGGER = logging.getLogger(__name__)


def disabled_train(self, mode=True):
    """Overwrite model.train with this function to make sure train/eval mode
    does not change anymore."""
    return self


def upload_to_s3(local_f, root_dir, client=None, destination=None, bucket=None):
    rel_path = str(pathlib.Path(local_f).absolute().relative_to(root_dir))
    s3_path = os.path.join(destination, rel_path)
    client.upload_file(local_f, bucket, s3_path)
    LOGGER.info(f"Upload {local_f} to {s3_path} on bucket {bucket}")


def download_from_s3(local_f, root_dir, client=None, destination=None, bucket=None):
    rel_path = str(pathlib.Path(local_f).absolute().relative_to(root_dir))
    s3_path = os.path.join(destination, rel_path)
    client.download_file(bucket, s3_path, local_f)
    LOGGER.info(f"Download {s3_path} on bucket {bucket} to {local_f}")


def save_ckpt(
    *,
    cfg,
    hydra_config,
    ckpt_dir,
    epoch,
    step_in_epoch,
    total_steps,
    total_steps_on_epoch_start,
    model,
    modules_not_to_save,
    optimizer=None,
    is_best=False,
    extra_info=None,
    ckpt_root_dir=None,
    s3_info=None,
):
    """Save the checkpoint."""
    if is_best:
        ckpt_path = os.path.join(ckpt_dir, "best.pth")
    else:
        ckpt_path = os.path.join(
            ckpt_dir, f"epoch_{epoch:06d}-step_{total_steps:09d}.pth"
        )
    if isinstance(model, DDP):
        model = model.module

    if modules_not_to_save is not None:
        assert isinstance(model, torch.nn.Module), f"{type(model)}"
        # filter out modules
        filtered_state_dict = []
        model_state_dict_to_save = {}
        for k in model.state_dict():
            flag_save = True
            for filter_k in modules_not_to_save:
                if k[: len(filter_k)] == filter_k:
                    flag_save = False
                    break
            if flag_save:
                model_state_dict_to_save[k] = model.state_dict()[k]
            else:
                filtered_state_dict.append(k)
    else:
        if isinstance(model, dict):
            model_state_dict_to_save = {}
            for k in model:
                model_state_dict_to_save[k] = model[k].state_dict()
        elif isinstance(model, torch.nn.Module):
            model_state_dict_to_save = model.state_dict()
        else:
            raise TypeError(type(model))

        filtered_state_dict = {}

    # LOGGER.info(f"Model params not saved: {filtered_state_dict}\n")
    LOGGER.info(f"Model modules not saved: {modules_not_to_save}\n")

    dict_to_save = {
        "cfg": cfg,
        "hydra_config": hydra_config,
        "epoch": epoch,
        "step_in_epoch": step_in_epoch,
        "total_steps": total_steps,
        "total_steps_on_epoch_start": total_steps_on_epoch_start,
        "model": model_state_dict_to_save,
        # "discriminator": discriminator_state_dict,
        # "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "extra_info": extra_info,
    }

    if optimizer is not None:
        if isinstance(optimizer, dict):
            dict_to_save["optimizer"] = {}
            for k in optimizer:
                dict_to_save["optimizer"][k] = optimizer[k].state_dict()
        elif isinstance(optimizer, torch.optim.Optimizer):
            dict_to_save["optimizer"] = optimizer.state_dict()
        else:
            raise TypeError(type(optimizer))

    torch.save(dict_to_save, ckpt_path)
    LOGGER.info("Save checkpoint to: %s" % ckpt_path)

    if s3_info is not None:
        upload_to_s3(ckpt_path, ckpt_root_dir, **s3_info)


def clean_up_ckpt(ckpt_dir, n_keep=5):
    """Clean up the checkpoints to keep only the last few (also keep the best one)."""
    if not os.path.exists(ckpt_dir):
        LOGGER.info(f"The following path does not exist: {ckpt_dir}\n")
        return
    ckpt_paths = sorted(
        [
            pathlib.Path(ckpt_dir) / fp
            for fp in os.listdir(ckpt_dir)
            if (".pth" in fp and "best.pth" not in fp)
        ]
    )
    if len(ckpt_paths) > n_keep:
        for ckpt_path in ckpt_paths[:-n_keep]:
            LOGGER.warning("Remove checkpoint: %s" % ckpt_path)
            os.remove(ckpt_path)


def get_ckpt_path(ckpt_dir, epoch=None):
    if not os.path.exists(ckpt_dir):
        LOGGER.info(f"The following path does not exist: {ckpt_dir}\n")
        return 0
    if epoch is not None:
        ckpt_paths = list(
            pathlib.Path(ckpt_dir).glob(f"epoch_{epoch:06d}*.pth")
        ) + list(pathlib.Path(ckpt_dir).glob(f"checkpoints/epoch_{epoch:06d}*.pth"))
        assert len(ckpt_paths) == 1, f"{ckpt_dir}, {len(ckpt_paths)}"
        assert ckpt_paths[0].exists(), f"{ckpt_dir}, {ckpt_paths[0]}"
    elif os.path.exists(os.path.join(ckpt_dir, "best.pth")):
        ckpt_paths = [os.path.join(ckpt_dir, "best.pth")]
    else:
        ckpt_paths = sorted(
            list(pathlib.Path(ckpt_dir).glob(f"*.pth"))
            + list(pathlib.Path(ckpt_dir).glob(f"checkpoints/*.pth"))
        )
    return ckpt_paths


def resume_one_nn_module(
    *, model, ckpt_data, strict, modules_not_saved, model_name="Model"
):
    tgt_state_dict = {
        key.replace("module.", ""): value for key, value in ckpt_data.items()
    }

    missed_keys, unexpcted_keys = model.load_state_dict(tgt_state_dict, strict=strict)

    assert len(unexpcted_keys) == 0, f"{unexpcted_keys}"
    missed_keys_not_in_not_saved_modules = []
    if modules_not_saved is not None:
        for k in missed_keys:
            flag_weird = True
            for filtered_k in modules_not_saved:
                if k[: len(filtered_k)] == filtered_k:
                    flag_weird = False
                    break
            if flag_weird:
                missed_keys_not_in_not_saved_modules.append(k)

    LOGGER.info(f"[{model_name} resuming] Modules not saved: {modules_not_saved}")
    LOGGER.info(
        f"[{model_name} resuming] Missed keys not in modules not saved: {missed_keys_not_in_not_saved_modules}"
    )


def resume_from_ckpt(
    *,
    ckpt_dir,
    model,
    modules_not_saved,
    optimizer=None,
    epoch=None,
    strict=True,
    cfg=None,
    device=torch.device("cpu"),
):
    """Resume the model & optimizer from the latest/specific checkpoint.

    Return:
        the epoch of the ckpt. return 0 if no ckpt found.
    """
    ckpt_paths = get_ckpt_path(ckpt_dir, epoch=epoch)
    LOGGER.info(f"ckpt paths: {ckpt_paths}")
    if len(ckpt_paths) > 0:
        ckpt_path = ckpt_paths[-1]
        ckpt_data = torch.load(ckpt_path, map_location=device)

        if isinstance(model, torch.nn.Module):
            resume_one_nn_module(
                model=model.module if isinstance(model, DDP) else model,
                ckpt_data=ckpt_data["model"],
                strict=strict,
                modules_not_saved=modules_not_saved,
                model_name="Model",
            )
            if optimizer is not None:
                optimizer.load_state_dict(ckpt_data["optimizer"])
        elif isinstance(model, dict):
            key_list = list(model.keys())
            for k in key_list:
                resume_one_nn_module(
                    model=model[k].module if isinstance(model[k], DDP) else model[k],
                    ckpt_data=ckpt_data["model"][k],
                    strict=strict,
                    modules_not_saved=modules_not_saved,
                    model_name=f"Model {k}",
                )
                if optimizer is not None:
                    optimizer[k].load_state_dict(ckpt_data["optimizer"][k])
        else:
            raise TypeError(type(model))

        epoch = ckpt_data["epoch"]
        step_in_epoch = ckpt_data["step_in_epoch"]
        total_steps = ckpt_data["total_steps"]
        total_steps_on_epoch_start = ckpt_data["total_steps_on_epoch_start"]

        LOGGER.info("[Model resuming] Load model from checkpoint: %s" % ckpt_path)
    else:
        epoch = 0
        step_in_epoch = 0
        total_steps = 0
        total_steps_on_epoch_start = 0
    return epoch, step_in_epoch, total_steps, total_steps_on_epoch_start


def learning_rate_decay(
    step, lr_init, lr_final, max_steps, lr_delay_steps=0, lr_delay_mult=1
):
    """Continuous learning rate decay function.
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    Args:
      step: int, the current optimization step.
      lr_init: float, the initial learning rate.
      lr_final: float, the final learning rate.
      max_steps: int, the number of steps during optimization.
      lr_delay_steps: int, the number of steps to delay the full learning rate.
      lr_delay_mult: float, the multiplier on the rate when delaying it.
    Returns:
      lr: the learning for current step 'step'.
    """
    if lr_delay_steps > 0:
        # A kind of reverse cosine decay.
        delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
            0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
        )
    else:
        delay_rate = 1.0
    t = np.clip(step / max_steps, 0, 1)
    log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
    return delay_rate * log_lerp


def calculate_psnr(img1, img2, mask):
    """
    https://github.com/google/dynibar/blob/02b164144cce2d93aa4c5d87b418497286b2ae31/eval_nvidia.py#L201

    Compute PSNR between two images.

    Args:
      img1: image 1
      img2: image 2
      mask: mask indicating which region is valid.

    Returns:
      PSNR: PSNR error
    """

    assert img1.ndim == 3, f"{img1.shape}"
    assert img2.ndim == 3, f"{img2.shape}"
    assert np.min(img1) >= 0 and np.max(img1) <= 1, f"{np.min(img1)}, {np.max(img1)}"
    assert np.min(img2) >= 0 and np.max(img2) <= 1, f"{np.min(img2)}, {np.max(img2)}"

    # img1 and img2 have range [0, 1]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mask = mask.astype(np.float64)

    num_valid = np.sum(mask) + 1e-8

    mse = np.sum((img1 - img2) ** 2 * mask) / num_valid

    if mse == 0:
        return 0  # float('inf')

    return 10 * math.log10(1.0 / mse)


def calculate_ssim(img1, img2, mask):
    """
    https://github.com/google/dynibar/blob/02b164144cce2d93aa4c5d87b418497286b2ae31/eval_nvidia.py#L228

    Compute SSIM between two images.

    Args:
      img1: image 1
      img2: image 2
      mask: mask indicating which region is valid.

    Returns:
      PSNR: PSNR error
    """
    assert img1.ndim == 3, f"{img1.shape}"
    assert img2.ndim == 3, f"{img2.shape}"
    assert np.min(img1) >= 0 and np.max(img1) <= 1, f"{np.min(img1)}, {np.max(img1)}"
    assert np.min(img2) >= 0 and np.max(img2) <= 1, f"{np.min(img2)}, {np.max(img2)}"

    if img1.shape != img2.shape:
        raise ValueError("Input images must have the same dimensions.")

    _, ssim_map = skimage.metrics.structural_similarity(
        img1, img2, full=True, channel_axis=2, data_range=2.0
    )

    # print("\nssim interal: ", _, np.mean(ssim_map), ssim_map.shape, "\n")

    num_valid = np.sum(mask) + 1e-8

    return np.sum(ssim_map * mask) / num_valid
