import logging
from typing import List

import torch

from pgdvs.models.gnt.models.transformer_network import GNT
from pgdvs.models.gnt.models.feature_network import ResUNet

LOGGER = logging.getLogger(__name__)


class GNTModel(torch.nn.Module):
    def __init__(
        self,
        *,
        netwidth,
        transformer_depth,
        coarse_feat_dim,
        fine_feat_dim,
        single_net,
        posenc_max_freq_log2,
        pos_enc_n_freqs,
        view_enc_n_freqs,
        ckpt_path=None,
    ):
        super(GNTModel, self).__init__()

        # create coarse GNT
        self.net_coarse = GNT(
            netwidth=netwidth,
            transformer_depth=transformer_depth,
            in_feat_ch=coarse_feat_dim,
            posenc_max_freq_log2=posenc_max_freq_log2,
            pos_enc_n_freqs=pos_enc_n_freqs,
            view_enc_n_freqs=view_enc_n_freqs,
            ret_alpha=True,
        )

        # single_net - trains single network which can be used for both coarse and fine sampling
        self.single_net = single_net
        if single_net:
            self.net_fine = None
        else:
            self.net_fine = GNT(
                netwidth=netwidth,
                transformer_depth=transformer_depth,
                in_feat_ch=fine_feat_dim,
                posenc_max_freq_log2=posenc_max_freq_log2,
                pos_enc_n_freqs=pos_enc_n_freqs,
                view_enc_n_freqs=view_enc_n_freqs,
                ret_alpha=True,
            )

        # create feature extraction network
        self.feature_net = ResUNet(
            coarse_out_ch=coarse_feat_dim,
            fine_out_ch=fine_feat_dim,
            single_net=single_net,
        )

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=[])

    def init_from_ckpt(self, path: str, ignore_keys: List[str] = list()):
        ckpt_data = torch.load(path, map_location="cpu")

        tgt_state_dict = {}
        for m_name in ["net_coarse", "feature_net", "net_fine"]:
            if m_name in ckpt_data:
                for k in ckpt_data[m_name]:
                    tgt_state_dict[f"{m_name}.{k}"] = ckpt_data[m_name][k]

        keys = list(tgt_state_dict.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    LOGGER.info("Deleting key {} from state_dict.".format(k))
                    del tgt_state_dict[k]

        LOGGER.info(f"[GNT] Loading from {path}")

        if len(ignore_keys) > 0:
            flag_strict = False
        else:
            flag_strict = True

        missed_keys, unexpcted_keys = self.load_state_dict(
            tgt_state_dict, strict=flag_strict
        )

        LOGGER.info(f"[GNT] Done loading from {path}")

        if len(ignore_keys) > 0:
            assert set(ignore_keys) == set(missed_keys), f"{ignore_keys}, {missed_keys}"

        if len(missed_keys) > 0:
            LOGGER.info("[GNT] Missing keys:")
            LOGGER.info(missed_keys)
        if len(unexpcted_keys) > 0:
            LOGGER.info("[GNT] Unexpected keys:")
            LOGGER.info(unexpcted_keys)
