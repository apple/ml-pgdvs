import logging
import pathlib
import numpy as np

from torch.utils.data import Dataset

from pgdvs.datasets.nvidia_eval import NvidiaDynEvaluationDataset
from pgdvs.datasets.nvidia_eval_pure_geo import NvidiaDynPureGeoEvaluationDataset
from pgdvs.datasets.nvidia_vis import NvidiaDynVisualizationDataset
from pgdvs.datasets.dycheck_iphone_eval import DyCheckiPhoneEvaluationDataset
from pgdvs.datasets.mono_vis import MonoVisualizationDataset


DEBUG_DIR = pathlib.Path(__file__).absolute().parent.parent.parent / "debug"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)


DATASET_DICT = {
    "nvidia_eval": NvidiaDynEvaluationDataset,
    "nvidia_eval_pure_geo": NvidiaDynPureGeoEvaluationDataset,
    "nvidia_vis": NvidiaDynVisualizationDataset,
    "dycheck_iphone_eval": DyCheckiPhoneEvaluationDataset,
    "mono_vis": MonoVisualizationDataset,
}


LOGGER = logging.getLogger(__name__)


class CombinedDataset(Dataset):
    def __init__(
        self,
        *,
        data_root,
        dataset_list,
        mode="train",
        max_hw=-1,
        rgb_range="0_1",
        use_aug=False,
        dataset_specifics={},
    ):
        if mode in ["eval", "vis"]:
            use_aug = False

        assert mode in ["train", "eval", "vis"], mode
        cur_dataset_list = dataset_list[mode]

        self.datasets = {}
        for dataset_name in cur_dataset_list:
            self.datasets[dataset_name] = DATASET_DICT[dataset_name](
                data_root=data_root,
                max_hw=max_hw,
                rgb_range=rgb_range,
                use_aug=use_aug,
                mode=mode,
                **dataset_specifics[dataset_name],
            )

        data_cnt = 0
        self.data_idxs = []
        sorted_dataset_names = sorted(
            list(self.datasets.keys())
        )  # to ensure all workers have the same order
        for tmp_name in sorted_dataset_names:
            for tmp_i in range(len(self.datasets[tmp_name])):
                self.data_idxs.append((data_cnt, tmp_name, tmp_i))
                data_cnt = data_cnt + 1

    def __len__(self):
        cur_len = np.sum([len(self.datasets[_]) for _ in self.datasets])
        assert cur_len == len(self.data_idxs), f"{cur_len}, {len(self.data_idxs)}"
        return cur_len

    def __getitem__(self, index):
        global_i, dataset_name, dataset_i = self.data_idxs[index]
        assert index == global_i, f"{index}, {global_i}"

        ret_dict = self.datasets[dataset_name][dataset_i]

        return ret_dict
