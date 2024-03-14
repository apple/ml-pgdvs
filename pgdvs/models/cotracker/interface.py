# Modified from https://github.com/facebookresearch/co-tracker/blob/0a0596b277545625054cb041f00419bcd3693ea5/demo.py
import tqdm
import logging

import torch

from pgdvs.models.cotracker.predictor import CoTrackerPredictor
from pgdvs.utils.rendering import modify_rgb_range


LOGGER = logging.getLogger(__name__)


class CoTrackerInterface(torch.nn.Module):
    def __init__(
        self,
        ckpt_path,
        ori_rgb_range="0_1",
        query_chunk_size=4096,
        local_rank=0,
    ):
        super().__init__()
        self.ori_rgb_range = ori_rgb_range
        self.query_chunk_size = query_chunk_size

        self.model = CoTrackerPredictor(checkpoint=ckpt_path)

        LOGGER.info(f"[CoTracker] Done loading checkpoint from {ckpt_path}")

    def __call__(self, *, frames, query_points):
        # frames: [N, H, W, 3]
        # query_points: [#pt, 3], 3 for [time, row, col]

        frames = frames.permute(0, 3, 1, 2)[None, ...]  # [1, N, 3, H, W]

        frames = modify_rgb_range(
            frames,
            src_range=self.ori_rgb_range,
            tgt_range="0_255",
            check_range=False,
            enforce_range=True,
        )

        # query points are formulated as [t, x, y]
        # https://github.com/facebookresearch/co-tracker/blob/0a0596b277545625054cb041f00419bcd3693ea5/cotracker/predictor.py#L37
        query_points = query_points[
            None, :, [0, 2, 1]
        ]  # [1, #pt, 3]; [t, row, col] -> [t, x, y]

        pred_tracks = []
        pred_visibility = []

        n_queries = query_points.shape[1]

        for start_i in tqdm.tqdm(
            range(0, n_queries, self.query_chunk_size), disable=True
        ):
            end_i = min(n_queries, start_i + self.query_chunk_size)
            tmp_queries = query_points[:, start_i:end_i, :]
            tmp_pred_tracks, tmp_pred_visibility = self.model(
                frames,
                queries=tmp_queries,
                grid_size=0,
                grid_query_frame=0,
                backward_tracking=False,
                segm_mask=None,
            )  # tracks: [1, #frame, #pt, 2], float32; visibility: [1, #frame, #pt], bool

            pred_tracks.append(tmp_pred_tracks)
            pred_visibility.append(tmp_pred_visibility)

        pred_tracks = torch.cat(pred_tracks, dim=2)[0, ...].permute(
            1, 0, 2
        )  # [1, #frame, #pt, 2] -> [#pt, #frame, 2]
        pred_visibility = torch.cat(pred_visibility, dim=2)[0, ...].permute(
            1, 0
        )  # [1, #frame, #pt] -> [#pt, #frame]

        # there may be negative values and we clip them
        pred_tracks = torch.clip(pred_tracks, 0.0)

        return pred_tracks, pred_visibility
