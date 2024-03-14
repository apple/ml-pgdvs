import io
import copy
import logging
import zipfile
import trimesh
import PIL.Image
import numpy as np
from scipy.spatial.transform import Rotation

import torch
from torch.utils.data import Dataset


LOGGER = logging.getLogger(__name__)


class BaseDataset(Dataset):
    aug_types = [
        "none",
        "rot90",
        "rot180",
        "rot270",
        # "flip_horizontal",
        # "flip_vertical",
    ]

    def _get_zip_obj(self):
        if self.zip_obj is None:
            # The open/close overhead is heavy, needs to have the handler always open
            self.zip_obj = zipfile.ZipFile(self.data_f)

    def __getitem__(self, index):
        raise NotImplementedError

    def augment_img(self, img, aug="none"):
        # rotation is anti clockwise
        if aug == "none":
            pass
        elif aug == "rot90":
            img = np.rot90(np.array(img))
        elif aug == "rot180":
            img = np.rot90(np.array(img), k=2)
        elif aug == "rot270":
            img = np.rot90(np.array(img), k=3)
        elif aug == "flip_horizontal":
            img = np.flip(np.array(img), axis=1)
        elif aug == "flip_vertical":
            img = np.flip(np.array(img), axis=0)
        else:
            raise ValueError(aug)
        return img

    def augment_flow(self, raw_flow, aug="none"):
        h, w, _ = raw_flow.shape

        # +U right, +V down
        # rotation is anti clockwise
        if aug == "none":
            new_flow = np.copy(raw_flow)
        elif aug == "rot90":
            # original pixel location (u1, v1) -> new location (v1, w - u1)
            # original flow (u2 - u1, v2 - v1) -> (v2 - v1, u1 - u2)
            new_flow = np.zeros((h, w, 2))
            new_flow[..., 0] = np.copy(raw_flow[..., 1])
            new_flow[..., 1] = np.copy(-1 * raw_flow[..., 0])
            new_flow = np.ascontiguousarray(np.rot90(new_flow))
        elif aug == "rot180":
            # original pixel location (u1, v1) -> new location (w - u1, h - v1)
            # original flow (u2 - u1, v2 - v1) -> (u1 - u2, v1 - v2)
            new_flow = np.zeros((h, w, 2))
            new_flow[..., 0] = np.copy(-1 * raw_flow[..., 0])
            new_flow[..., 1] = np.copy(-1 * raw_flow[..., 1])
            new_flow = np.ascontiguousarray(np.rot90(new_flow, k=2))
        elif aug == "rot270":
            # original pixel location (u1, v1) -> new location (h - v1, u1)
            # original flow (u2 - u1, v2 - v1) -> (v1 - v2, u2 - u1)
            new_flow = np.zeros((h, w, 2))
            new_flow[..., 0] = np.copy(-1 * raw_flow[..., 1])
            new_flow[..., 1] = np.copy(raw_flow[..., 0])
            new_flow = np.ascontiguousarray(np.rot90(new_flow, k=3))
        elif aug == "flip_horizontal":
            # original pixel location (u1, v1) -> new location (w - u1, v1)
            # original flow (u2 - u1, v2 - v1) -> (u1 - u2, v2 - v1)
            new_flow = np.zeros((h, w, 2))
            new_flow[..., 0] = np.copy(-1 * raw_flow[..., 0])
            new_flow[..., 1] = np.copy(raw_flow[..., 1])
            new_flow = np.ascontiguousarray(np.flip(new_flow, axis=1))
        elif aug == "flip_vertical":
            # original pixel location (u1, v1) -> new location (u1, h - v1)
            # original flow (u2 - u1, v2 - v1) -> (u2 - u1, v1 - v2)
            new_flow = np.zeros((h, w, 2))
            new_flow[..., 0] = np.copy(raw_flow[..., 0])
            new_flow[..., 1] = np.copy(-1 * raw_flow[..., 1])
            new_flow = np.ascontiguousarray(np.flip(new_flow, axis=0))
        else:
            raise ValueError(aug)

        return new_flow

    def augment_cam(self, aug, c2w, K, H, W):
        # We assume OpenCV format: +X right, +Y down, +Z forward
        w2c = copy.deepcopy(np.linalg.inv(c2w))
        K = copy.deepcopy(K)  # +X right, +Y down. [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]

        fx = K[0, 0]
        fy = K[1, 1]
        cx = K[0, 2]
        cy = K[1, 2]

        if aug == "none":
            pass
        elif aug == "rot90":
            # Since +Z forward, the angle rotation should be clockwise, i.e., negative.
            rot_mat = np.array(
                Rotation.from_rotvec(-np.pi / 2 * np.array([0, 0, 1])).as_matrix()
            )
            transform_mat = np.eye(4)
            transform_mat[:3, :3] = rot_mat
            w2c = np.matmul(transform_mat, w2c)
            # [[fy, 0, cy], [0, fx, W - cx], [0, 0, 1]]
            K = np.array([[fy, 0, cy], [0, fx, W - cx], [0, 0, 1]])
        elif aug == "rot180":
            rot_mat = np.array(
                Rotation.from_rotvec(-np.pi * np.array([0, 0, 1])).as_matrix()
            )
            transform_mat = np.eye(4)
            transform_mat[:3, :3] = rot_mat
            w2c = np.matmul(transform_mat, w2c)
            # [[fx, 0, W - cx], [0, fy, H - cy], [0, 0, 1]]
            K = np.array([[fx, 0, W - cx], [0, fy, H - cy], [0, 0, 1]])
        elif aug == "rot270":
            rot_mat = np.array(
                Rotation.from_rotvec(-np.pi * 1.5 * np.array([0, 0, 1])).as_matrix()
            )
            transform_mat = np.eye(4)
            transform_mat[:3, :3] = rot_mat
            w2c = np.matmul(transform_mat, w2c)
            # [[fy, 0, H - cy], [0, fx, cx], [0, 0, 1]]
            K = np.array([[fy, 0, H - cy], [0, fx, cx], [0, 0, 1]])
        elif aug == "flip_horizontal":
            if True:
                # NOTE: negative focal means camera caputres scenes behind the camera, which does not make sense.
                raise NotImplementedError
            else:
                # [[-fx, 0, W - cx], [0, fy, cy], [0, 0, 1]]
                K = np.array([[-fx, 0, W - cx], [0, fy, cy], [0, 0, 1]])
        elif aug == "flip_vertical":
            if True:
                # NOTE: negative focal means camera caputres scenes behind the camera, which does not make sense.
                raise NotImplementedError
            else:
                # [[fx, 0, cx], [0, -fy, H - cy], [0, 0, 1]]
                K = np.array([[fx, 0, cx], [0, -fy, H - cy], [0, 0, 1]])
        else:
            raise ValueError(aug)

        return np.linalg.inv(w2c), K

    def make_input_square(self, in_data, max_hw, use_aug=False, pad_info=None):
        cur_h, cur_w, _ = in_data.shape

        if cur_h == cur_w:
            # already square
            mask = np.ones((max_hw, max_hw, 1), dtype=bool)
            return in_data, mask, (0, 0)

        flow_square = np.zeros((max_hw, max_hw, in_data.shape[2]), dtype=np.float32)
        mask = np.zeros((max_hw, max_hw, 1), dtype=bool)

        pad_left = pad_top = 0
        # pad_right = cur_w
        # pad_bottom = cur_h

        if use_aug:
            if pad_info is None:
                pad_h = max_hw - cur_h
                pad_w = max_hw - cur_w
                if pad_w > 0:
                    pad_left = np.random.randint(pad_w)
                    # pad_right = pad_w - pad_left
                if pad_h > 0:
                    pad_top = np.random.randint(pad_h)
                    # pad_bottom = pad_h - pad_top
            else:
                pad_left, pad_top = pad_info

        flow_square[
            pad_top : (pad_top + cur_h), pad_left : (pad_left + cur_w), :
        ] = in_data
        mask[pad_top : (pad_top + cur_h), pad_left : (pad_left + cur_w), :] = True

        pad_info = (pad_left, pad_top)

        return flow_square, mask, pad_info

    def _resize_torch(self, x, tgt_h, tgt_w):
        return (
            torch.nn.functional.interpolate(
                torch.FloatTensor(x)[None, ...].permute(0, 3, 1, 2),
                size=(tgt_h, tgt_w),
                mode="bilinear",
                # mode="nearest",
                align_corners=True,
            )[0, ...]
            .permute(1, 2, 0)
            .contiguous()
            .numpy()
        )

    def _read_flow(self, zip_obj, flow_f):
        with io.BufferedReader(zip_obj.open(flow_f, mode="r")) as f:
            # - flow/coord_diff: [H, W, 2]
            # - ori_h, ori_w: [H, W, 2]
            flow_info = np.load(f, allow_pickle=True)
            raw_flow = flow_info["flow"]
        return raw_flow

    def process_flow_f(
        self, zip_obj, src_flow_f, tgt_flow_f=None, flow_res_agnostic=False
    ):
        raw_src_flow = self._read_flow(zip_obj, src_flow_f)

        if tgt_flow_f is None:
            raw_tgt_flow = np.copy(raw_src_flow)
        else:
            raw_tgt_flow = self._read_flow(zip_obj, tgt_flow_f)

        return self.process_flow(
            raw_src_flow, raw_tgt_flow, flow_res_agnostic=flow_res_agnostic
        )

    def process_flow(
        self,
        raw_flow,
        aug_type=None,
        square_pad_info=None,
        need_crop=None,
        crop_info=None,
    ):
        assert need_crop is None, f"{need_crop}"

        if self.use_aug:
            if aug_type is None:
                aug_type = np.random.choice(self.aug_types)
            else:
                assert aug_type in self.aug_types, f"{aug_type}"

            aug_flow = self.augment_flow(raw_flow, aug_type)
        else:
            aug_type = "none"
            aug_flow = raw_flow

        if self.max_hw > 0:
            flow, flow_mask, flow_pad_info = self.make_input_square(
                aug_flow, self.max_hw, use_aug=self.use_aug, pad_info=square_pad_info
            )
        else:
            # use raw resolution
            flow = aug_flow
            flow_mask = np.ones_like(flow[:, :, :1]).astype(bool)
            flow_pad_info = (0, 0)

        extra_outs = {
            "aug_type": aug_type,
            "pad_info": flow_pad_info,
            "raw_h": raw_flow.shape[0],
            "raw_w": raw_flow.shape[1],
            "aug_h": aug_flow.shape[0],
            "aug_w": aug_flow.shape[1],
            "aug_flow": aug_flow,
            "crop_info": crop_info,
        }

        return flow, flow_mask, extra_outs

    def _imread(self, zip_obj, f):
        img = np.array(PIL.Image.open(zip_obj.open(f)))
        if img.ndim == 2:
            # E.g., ImageNet, ILSVRC2012_img_test_v10102019/ILSVRC2012_test_00000073.JPEG
            img = np.tile(img[..., None], 3)
        if img.shape[2] == 4:
            img = img[..., :3]
        return img

    def _crop_img(self, raw_img, crop_type, crop_size, crop_info=None):
        raw_h, raw_w, _ = raw_img.shape

        if crop_info is None:
            crop_h, crop_w = crop_size
            assert crop_h <= raw_h, f"{raw_h}, {crop_h}"
            assert crop_w <= raw_w, f"{raw_w}, {crop_w}"

            if crop_type == "center":
                pad_top = int(round((raw_h - crop_h) / 2.0))
                pad_left = int(round((raw_w - crop_w) / 2.0))
            elif crop_type == "rnd":
                pad_top, pad_left = 0, 0
                space_h = raw_h - crop_h
                space_w = raw_w - crop_w
                if space_w > 0:
                    pad_left = np.random.randint(space_w)
                if space_h > 0:
                    pad_top = np.random.randint(space_h)
            else:
                raise ValueError(crop_type)

            h_start = pad_top
            h_end = pad_top + crop_h
            w_start = pad_left
            w_end = pad_left + crop_w

            crop_info = {
                "h_start": h_start,
                "h_end": h_end,
                "w_start": w_start,
                "w_end": w_end,
                "crop_h": crop_h,
                "crop_w": crop_w,
            }
        else:
            h_start = crop_info["h_start"]
            h_end = crop_info["h_end"]
            w_start = crop_info["w_start"]
            w_end = crop_info["w_end"]
            crop_h = crop_info["crop_h"]
            crop_w = crop_info["crop_w"]

            assert crop_h <= raw_h, f"{raw_h}, {crop_h}"
            assert crop_w <= raw_w, f"{raw_w}, {crop_w}"

        cropped_img = raw_img[h_start:h_end, w_start:w_end, :]

        return cropped_img, crop_info

    def process_img_f(
        self,
        zip_obj,
        img_f,
        aug_type=None,
        square_pad_info=None,
        tgt_shape=None,
        need_crop=None,
        crop_info=None,
    ):
        raw_img = self._imread(zip_obj, img_f)
        if tgt_shape is not None:
            tgt_h, tgt_w = tgt_shape
            if raw_img.shape[0] != tgt_h or raw_img.shape[1] != tgt_w:
                raw_img = np.array(
                    PIL.Image.fromarray(raw_img).resize(
                        (tgt_w, tgt_h), resample=PIL.Image.Resampling.LANCZOS
                    )
                )

        img, img_mask, extra_outs = self.process_img(
            raw_img,
            aug_type=aug_type,
            square_pad_info=square_pad_info,
            need_crop=need_crop,
            crop_info=crop_info,
        )

        return img, img_mask, extra_outs

    def process_img(
        self,
        raw_img,
        aug_type=None,
        square_pad_info=None,
        need_crop=None,
        crop_info=None,
    ):
        if need_crop is not None:
            assert need_crop in ["rnd", "center"], f"{need_crop}"
            raw_img, crop_info = self._crop_img(
                raw_img, need_crop, (self.max_hw, self.max_hw), crop_info=crop_info
            )

        if self.use_aug:
            if aug_type is None:
                aug_type = np.random.choice(self.aug_types)
            else:
                assert aug_type in self.aug_types, f"{aug_type}"

            aug_img = self.augment_img(raw_img, aug_type)
        else:
            aug_type = "none"
            aug_img = raw_img

        if self.max_hw > 0:
            img, img_mask, img_pad_info = self.make_input_square(
                aug_img, self.max_hw, use_aug=self.use_aug, pad_info=square_pad_info
            )
        else:
            # use raw resolution
            img = aug_img
            img_mask = np.ones_like(img[:, :, :1]).astype(bool)
            img_pad_info = (0, 0)

        extra_outs = {
            "aug_type": aug_type,
            "pad_info": img_pad_info,
            "raw_h": raw_img.shape[0],
            "raw_w": raw_img.shape[1],
            "aug_h": aug_img.shape[0],
            "aug_w": aug_img.shape[1],
            "aug_img": aug_img,
            "crop_info": crop_info,
        }

        return img, img_mask, extra_outs

    def sort_poses_wrt_ref(
        self,
        *,
        tgt_pose,
        ref_poses,
        tgt_id=-1,
        dist_method="vector",
        scene_center=(0, 0, 0),
    ):
        """
        Args:
            tgt_pose: target pose [4, 4]
            ref_poses: reference poses [N, 4, 4]
            num_select: the number of nearest views to select
        Returns: the selected indices
        """
        num_cams = len(ref_poses)
        batched_tgt_pose = tgt_pose[None, ...].repeat(num_cams, 0)  # [N, 4, 4]

        if dist_method == "matrix":
            dists = batched_angular_dist_rot_matrix(
                batched_tgt_pose[:, :3, :3], ref_poses[:, :3, :3]
            )
        elif dist_method == "vector":
            tar_cam_locs = batched_tgt_pose[:, :3, 3]
            ref_cam_locs = ref_poses[:, :3, 3]
            scene_center = np.array(scene_center)[None, ...]
            tar_vectors = tar_cam_locs - scene_center
            ref_vectors = ref_cam_locs - scene_center
            dists = angular_dist_between_2_vectors(tar_vectors, ref_vectors)
        elif dist_method == "dist":
            tar_cam_locs = batched_tgt_pose[:, :3, 3]
            ref_cam_locs = ref_poses[:, :3, 3]
            dists = np.linalg.norm(tar_cam_locs - ref_cam_locs, axis=1)
        elif dist_method == "dist_matrix":
            dists_1 = batched_angular_dist_rot_matrix(
                batched_tgt_pose[:, :3, :3], ref_poses[:, :3, :3]
            )

            min_dists_1 = np.min(dists_1)
            max_dists_1 = np.max(dists_1)
            dists_1 = (dists_1 - min_dists_1) / (max_dists_1 - min_dists_1 + 1e-8)

            tar_cam_locs = batched_tgt_pose[:, :3, 3]
            ref_cam_locs = ref_poses[:, :3, 3]
            dists_2 = np.linalg.norm(tar_cam_locs - ref_cam_locs, axis=1)

            min_dists_2 = np.min(dists_2)
            max_dists_2 = np.max(dists_2)
            dists_2 = (dists_2 - min_dists_2) / (max_dists_2 - min_dists_2 + 1e-8)

            dists = dists_1 + dists_2
        else:
            raise Exception("unknown angular distance calculation method!")

        if tgt_id >= 0:
            assert tgt_id < num_cams
            dists[tgt_id] = 99999999  # make sure not to select the target id itself

        sorted_ids = np.argsort(dists)
        sorted_dists = dists[sorted_ids]
        return sorted_ids, sorted_dists

    def modify_K_wrt_crop(self, K_normalized, raw_shape, crop_info):
        return modify_K_wrt_crop_func(K_normalized, raw_shape, crop_info, self.max_hw)

    def _vis_pcl(self, tgt_K, tgt_w2c, tgt_img, tgt_depth, save_f):
        tgt_h, tgt_w, _ = tgt_img.shape
        tgt_K_torch = torch.FloatTensor(tgt_K)[None, ...]
        tgt_c2w_torch = torch.FloatTensor(np.linalg.inv(tgt_w2c))[None, ...]
        tgt_rays_o, tgt_rays_d, tgt_rays_rgb, tgt_uvs = self._get_rays_single_image(
            tgt_h, tgt_w, tgt_K_torch, tgt_c2w_torch
        )  # [H x W, 3]
        # print(tgt_rays_o.shape, tgt_rays_d.shape, tgt_rays_rgb.shape)

        tgt_depth = tgt_depth.reshape((-1, 1))  # [H, W, 1]
        # print("tgt_depth: ", tgt_depth.shape)
        mesh_verts = tgt_rays_o + tgt_rays_d * tgt_depth

        mesh_colors = tgt_img.reshape((-1, 3))

        mesh_pcl = trimesh.PointCloud(
            vertices=mesh_verts, colors=mesh_colors, process=False
        )
        _ = mesh_pcl.export(save_f)

        extra_outs = {
            "uv": tgt_uvs,
            "ray_o": tgt_rays_o,
            "ray_d": tgt_rays_d,
        }

        return mesh_verts, mesh_colors, extra_outs

    def _get_rays_single_image(self, H, W, intrinsics, c2w):
        """
        :param H: image height
        :param W: image width
        :param intrinsics: 4 by 4 intrinsic matrix
        :param c2w: 4 by 4 camera to world extrinsic matrix
        :return:
        """
        batch_size = 1
        render_stride = 1

        u, v = np.meshgrid(np.arange(W)[::render_stride], np.arange(H)[::render_stride])

        base_rgb1 = np.array([1.0, 0, 0]).reshape((1, 1, 3))  # red
        base_rgb2 = np.array([0, 1.0, 0]).reshape((1, 1, 3))  # green
        # top-left red, bottom-right green, top-right black
        rgbs = (W - u[..., None]) / W * base_rgb1 + v[..., None] / H * base_rgb2
        # print("rgbs: ", rgbs.shape)
        rgbs = (rgbs * 255).reshape((-1, 3)).astype(np.uint8)

        u = u.reshape(-1).astype(dtype=np.float32)  # + 0.5    # add half pixel
        v = v.reshape(-1).astype(dtype=np.float32)  # + 0.5
        pixels = np.stack((u, v, np.ones_like(u)), axis=0)  # (3, H*W)
        pixels = torch.from_numpy(pixels)
        batched_pixels = pixels.unsqueeze(0).repeat(batch_size, 1, 1)

        rays_d = (
            c2w[:, :3, :3].bmm(torch.inverse(intrinsics[:, :3, :3])).bmm(batched_pixels)
        ).transpose(1, 2)
        rays_d = rays_d.reshape(-1, 3)
        rays_o = (
            c2w[:, :3, 3].unsqueeze(1).repeat(1, rays_d.shape[0], 1).reshape(-1, 3)
        )  # B x HW x 3

        return (
            rays_o.numpy(),
            rays_d.numpy(),
            rgbs,
            pixels.permute(1, 0).contiguous().numpy(),
        )


def modify_K_wrt_crop_func(K_normalized, raw_shape, crop_info, tgt_crop_hw):
    # NOTE: since we may alraedy change resolution of rgb,
    # we need to get the raw intrinsics from the normalized one.
    # h != rgb.shape[0] here since we have already called self._aug_rgb
    raw_h, raw_w = raw_shape
    raw_K = copy.deepcopy(K_normalized)
    raw_K[0, :] = raw_K[0, :] * raw_w
    raw_K[1, :] = raw_K[1, :] * raw_h

    raw_K[0, 2] = raw_K[0, 2] - crop_info["w_start"]
    raw_K[1, 2] = raw_K[1, 2] - crop_info["h_start"]

    assert crop_info["crop_w"] == tgt_crop_hw, f'{crop_info["crop_w"]}, {tgt_crop_hw}'
    assert crop_info["crop_h"] == tgt_crop_hw, f'{crop_info["crop_h"]}, {tgt_crop_hw}'

    K_normalized = copy.deepcopy(raw_K)
    K_normalized[0, :] = K_normalized[0, :] / crop_info["crop_w"]
    K_normalized[1, :] = K_normalized[1, :] / crop_info["crop_h"]

    return K_normalized


TINY_NUMBER = 1e-6  # float32 only has 7 decimal digits precision


def angular_dist_between_2_vectors(vec1, vec2):
    vec1_unit = vec1 / (np.linalg.norm(vec1, axis=1, keepdims=True) + TINY_NUMBER)
    vec2_unit = vec2 / (np.linalg.norm(vec2, axis=1, keepdims=True) + TINY_NUMBER)
    angular_dists = np.arccos(
        np.clip(np.sum(vec1_unit * vec2_unit, axis=-1), -1.0, 1.0)
    )
    return angular_dists


def batched_angular_dist_rot_matrix(R1, R2):
    """
    calculate the angular distance between two rotation matrices (batched)
    :param R1: the first rotation matrix [N, 3, 3]
    :param R2: the second rotation matrix [N, 3, 3]
    :return: angular distance in radiance [N, ]
    """
    assert (
        R1.shape[-1] == 3
        and R2.shape[-1] == 3
        and R1.shape[-2] == 3
        and R2.shape[-2] == 3
    )
    return np.arccos(
        np.clip(
            (np.trace(np.matmul(R2.transpose(0, 2, 1), R1), axis1=1, axis2=2) - 1)
            / 2.0,
            a_min=-1 + TINY_NUMBER,
            a_max=1 - TINY_NUMBER,
        )
    )
