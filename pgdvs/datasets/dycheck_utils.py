import json
import copy
import PIL.Image
import numpy as np
import os.path as osp
from typing import Optional, Sequence, Tuple, Union, Dict

from pgdvs.utils.dycheck.camera import DyCheckCamera


class iPhoneParser:
    """Parser for the iPhone dataset.

    Modified from
    - https://github.com/KAIR-BAIR/dycheck/blob/ddf77a4e006fdbc5aed28e0859c216da0de5aff5/dycheck/datasets/iphone.py#L931
    - https://github.com/KAIR-BAIR/dycheck/blob/ddf77a4e006fdbc5aed28e0859c216da0de5aff5/dycheck/datasets/nerfies.py#L99
    """

    SPLITS = ["train", "val"]

    def __init__(
        self,
        # dataset: str,
        sequence: str,
        *,
        data_root=None,
        use_undistort: bool = False,
    ):
        # self.dataset = dataset
        self.sequence = sequence
        self.data_root = data_root or osp.abspath(
            osp.join(osp.dirname(__file__), "..", "..", "datasets")
        )
        # self.data_dir = osp.join(self.data_root, self.dataset, self.sequence)
        self.data_dir = osp.join(self.data_root, self.sequence)

        self.use_undistort = False

        (
            self._center,
            self._scale,
            self._near,
            self._far,
        ) = self._load_scene_info(self.data_dir)
        (
            self._frame_names_map,
            self._time_ids,
            self._camera_ids,
        ) = self._load_metadata_info(self.data_dir)
        self._load_extra_info()

        self.splits_dir = osp.join(self.data_dir, "splits")
        if not osp.exists(self.splits_dir):
            self._create_splits()

        self.use_undistort = use_undistort
        assert not self.use_undistort

    def __len__(self):
        return len(self.time_ids)

    def load_split(self, split: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        assert split in self.SPLITS

        with open(osp.join(self.splits_dir, f"{split}.json")) as f:
            split_dict = json.load(f)
        return (
            np.array(split_dict["frame_names"]),
            np.array(split_dict["time_ids"], np.uint32),
            np.array(split_dict["camera_ids"], np.uint32),
        )

    def _load_scene_info(self, data_dir) -> Tuple[np.ndarray, float, float, float]:
        with open(osp.join(data_dir, "scene.json")) as f:
            scene_dict = json.load(f)
        center = np.array(scene_dict["center"], dtype=np.float32)
        scale = scene_dict["scale"]
        near = scene_dict["near"]
        far = scene_dict["far"]
        return center, scale, near, far

    def _load_metadata_info(
        self, data_dir
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        with open(osp.join(data_dir, "dataset.json")) as f:
            dataset_dict = json.load(f)
        _frame_names = np.array(dataset_dict["ids"])

        with open(osp.join(data_dir, "metadata.json")) as f:
            metadata_dict = json.load(f)
        time_ids = np.array(
            [metadata_dict[k]["warp_id"] for k in _frame_names], dtype=np.uint32
        )
        camera_ids = np.array(
            [metadata_dict[k]["camera_id"] for k in _frame_names], dtype=np.uint32
        )

        frame_names_map = np.zeros(
            (time_ids.max() + 1, camera_ids.max() + 1), _frame_names.dtype
        )
        for i, (t, c) in enumerate(zip(time_ids, camera_ids)):
            frame_names_map[t, c] = _frame_names[i]

        return frame_names_map, time_ids, camera_ids

    def load_rgba(
        self,
        time_id: int,
        camera_id: int,
        *,
        use_undistort: Optional[bool] = None,
    ) -> np.ndarray:
        if use_undistort is None:
            use_undistort = self.use_undistort

        frame_name = self._frame_names_map[time_id, camera_id]
        rgb_path = osp.join(
            self.data_dir,
            "rgb" if not use_undistort else "rgb_undistort",
            f"{self._factor}x",
            frame_name + ".png",
        )
        if osp.exists(rgb_path):
            # rgba = io.load(rgb_path, flags=cv2.IMREAD_UNCHANGED)
            rgba = np.array(PIL.Image.open(rgb_path))
            if rgba.shape[-1] == 3:
                rgba = np.concatenate([rgba, np.full_like(rgba[..., :1], 255)], axis=-1)
        elif use_undistort:
            raise NotImplementedError
            # camera = self.load_camera(time_id, camera_id, use_undistort=False)
            # rgb = self.load_rgba(time_id, camera_id, use_undistort=False)[
            #     ..., :3
            # ]
            # rgb = cv2.undistort(rgb, camera.intrin, camera.distortion)
            # alpha = (
            #     cv2.undistort(
            #         np.full_like(rgb, 255),
            #         camera.intrin,
            #         camera.distortion,
            #     )
            #     == 255
            # )[..., :1].astype(np.uint8) * 255
            # rgba = np.concatenate([rgb, alpha], axis=-1)
            # io.dump(rgb_path, rgba)
        else:
            raise ValueError(f"RGB image not found: {rgb_path}.")
        return rgba

    def load_depth(
        self,
        time_id: int,
        camera_id: int,
    ) -> np.ndarray:
        frame_name = self._frame_names_map[time_id, camera_id]

        depth_path = osp.join(
            self.data_dir, "depth", f"{self._factor}x", frame_name + ".npy"
        )
        depth = np.load(depth_path, allow_pickle=True) * self.scale
        # camera = self.load_camera(time_id, camera_id)
        # # The original depth data is projective; convert it to ray traveling
        # # distance.
        # depth = depth / camera.pixels_to_cosa(camera.get_pixels())
        return depth

    def load_camera(
        self,
        time_id: int,
        camera_id: int,
        *,
        use_undistort: Optional[bool] = None,
        **_,
    ):
        if use_undistort is None:
            use_undistort = self.use_undistort

        frame_name = self._frame_names_map[time_id, camera_id]
        camera = (
            DyCheckCamera.fromjson(
                osp.join(self.data_dir, "camera", frame_name + ".json")
            )
            .rescale_image_domain(1 / self._factor)
            .translate(-self._center)
            .rescale(self._scale)
        )
        if use_undistort:
            camera = camera.undistort_image_domain()
        return camera

    def load_covisible(
        self,
        time_id: int,
        camera_id: int,
        split: str,
        *,
        use_undistort: Optional[bool] = None,
        **_,
    ) -> np.ndarray:
        if use_undistort is None:
            use_undistort = self.use_undistort

        frame_name = self._frame_names_map[time_id, camera_id]
        covisible_path = osp.join(
            self.data_dir,
            "covisible" if not use_undistort else "covisible_undistort",
            f"{self._factor}x",
            split,
            frame_name + ".png",
        )
        if osp.exists(covisible_path):
            # (H, W, 1) uint8 mask.
            # covisible = io.load(covisible_path)[..., :1]
            covisible = np.array(PIL.Image.open(covisible_path))
        elif use_undistort:
            raise NotImplementedError
            # camera = self.load_camera(time_id, camera_id, use_undistort=False)
            # covisible = self.load_covisible(
            #     time_id,
            #     camera_id,
            #     split,
            #     use_undistort=False,
            # ).repeat(3, axis=-1)
            # alpha = (
            #     cv2.undistort(
            #         np.full_like(covisible, 255),
            #         camera.intrin,
            #         camera.distortion,
            #     )
            #     == 255
            # )[..., :1].astype(np.uint8) * 255
            # covisible = cv2.undistort(
            #     covisible, camera.intrin, camera.distortion
            # )[..., :1]
            # covisible = ((covisible == 255) & (alpha == 255)).astype(
            #     np.uint8
            # ) * 255
            # io.dump(covisible_path, covisible)
        else:
            raise ValueError(
                f"Covisible image not found: {covisible_path}. If not "
                f"processed before, please consider running "
                f"tools/process_covisible.py."
            )
        return covisible

    def _load_extra_info(self) -> None:
        extra_path = osp.join(self.data_dir, "extra.json")
        with open(extra_path) as f:
            extra_dict = json.load(f)
        self._factor = extra_dict["factor"]
        self._fps = extra_dict["fps"]
        self._bbox = np.array(extra_dict["bbox"], dtype=np.float32)
        self._lookat = np.array(extra_dict["lookat"], dtype=np.float32)
        self._up = np.array(extra_dict["up"], dtype=np.float32)

    def dump_json(
        self,
        filename,
        obj: Dict,
        *,
        sort_keys: bool = True,
        indent: Optional[int] = 4,
        separators: Tuple[str, str] = (",", ": "),
        **kwargs,
    ) -> None:
        # Process potential numpy arrays.
        if isinstance(obj, dict):
            obj = {k: v.tolist() if hasattr(v, "tolist") else v for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            pass
        elif isinstance(obj, np.ndarray):
            obj = obj.tolist()
        else:
            raise ValueError(f"{type(obj)} is not a supported type.")
        # Prefer visual appearance over compactness.
        with open(filename, "w") as f:
            json.dump(
                obj,
                f,
                sort_keys=sort_keys,
                indent=indent,
                separators=separators,
                **kwargs,
            )

    def _create_splits(self):
        def _create_split(split):
            assert split in self.SPLITS, f'Unknown split "{split}".'

            if split == "train":
                mask = self.camera_ids == 0
            elif split == "val":
                mask = self.camera_ids != 0
            else:
                raise ValueError(f"Unknown split {split}.")

            frame_names = self.frame_names[mask]
            time_ids = self.time_ids[mask]
            camera_ids = self.camera_ids[mask]
            split_dict = {
                "frame_names": frame_names,
                "time_ids": time_ids,
                "camera_ids": camera_ids,
            }
            self.dump_json(osp.join(self.splits_dir, f"{split}.json"), split_dict)

        # common.parallel_map(_create_split, self.SPLITS)
        for split in self.SPLITS:
            _create_split(split)

    @property
    def frame_names(self):
        return self._frame_names_map[self.time_ids, self.camera_ids]

    def get_frame_name(self, time_id, camera_id):
        return self._frame_names_map[time_id, camera_id]

    @property
    def time_ids(self):
        return self._time_ids

    @property
    def camera_ids(self):
        return self._camera_ids

    @property
    def center(self):
        return self._center

    @property
    def scale(self):
        return self._scale

    @property
    def near(self):
        return self._near

    @property
    def far(self):
        return self._far

    @property
    def factor(self):
        return self._factor

    @property
    def fps(self):
        return self._fps

    @property
    def bbox(self):
        return self._bbox

    @property
    def lookat(self):
        return self._lookat

    @property
    def up(self):
        return self._up
