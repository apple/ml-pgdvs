import copy
import cv2
import PIL.Image
import numpy as np

import torch
import torch.nn.functional as F


class Projector:
    def __init__(self):
        pass

    def inbound(self, pixel_locations, h, w):
        """
        check if the pixel locations are in valid range
        :param pixel_locations: [..., 2]
        :param h: height
        :param w: weight
        :return: mask, bool, [...]
        """
        return (
            (pixel_locations[..., 0] <= w - 1.0)
            & (pixel_locations[..., 0] >= 0)
            & (pixel_locations[..., 1] <= h - 1.0)
            & (pixel_locations[..., 1] >= 0)
        )

    def normalize(self, pixel_locations, h, w):
        n_dim = pixel_locations.ndim
        resize_factor = (
            torch.tensor([w - 1.0, h - 1.0])
            .to(pixel_locations.device)
            .reshape([1] * (n_dim - 1) + [2])
        )  # [1, 1, 2]
        normalized_pixel_locations = (
            2 * pixel_locations / resize_factor - 1.0
        )  # [#src, n_points, 2]
        return normalized_pixel_locations

    def compute_projections(self, xyz, train_cameras):
        """
        project 3D points into cameras
        :param xyz: [..., 3]
        :param train_cameras: [#src, 34], 34 = img_size(2) + intrinsics(16) + extrinsics(16)
        :return: pixel locations [..., 2], mask [...]
        """
        num_views = len(train_cameras)
        train_intrinsics = train_cameras[:, 2:18].reshape(-1, 4, 4)  # [#src, 4, 4]
        train_poses = train_cameras[:, -16:].reshape(-1, 4, 4)  # [#src, 4, 4]
        if xyz.ndim == 3:
            # here xyz: [#ray, #sample, 3]
            original_shape = xyz.shape[:2]  # [#ray, #sample]
            xyz = xyz.reshape(-1, 3)  # [#pts, 3]
            xyz_h = torch.cat(
                [xyz, torch.ones_like(xyz[..., :1])], dim=-1
            )  # [n_points, 4]
            projections = train_intrinsics.bmm(torch.inverse(train_poses)).bmm(
                xyz_h.t()[None, ...].repeat(num_views, 1, 1)
            )  # [#src, 4, n_points] = [#src, 4, 4] x [#src, 4, 4] x [#src, 4, #pts]
        else:
            raise AttributeError(xyz.shape)
        projections = projections.permute(0, 2, 1)  # [#src, n_points, 4]
        pixel_locations = projections[..., :2] / torch.clamp(
            projections[..., 2:3], min=1e-8
        )  # [#src, n_points, 2]
        pixel_locations = torch.clamp(pixel_locations, min=-1e6, max=1e6)
        mask = projections[..., 2] > 0  # a point is invalid if behind the camera
        pixel_locations = pixel_locations.reshape(
            (num_views,) + original_shape + (2,)
        )  # [#src, #ray, #sample, 2]
        mask = mask.reshape((num_views,) + original_shape)  # [#src, #ray, #sample]
        return pixel_locations, mask

    def compute_angle(self, xyz, query_camera, train_cameras):
        """
        :param xyz: [..., 3].
        :param query_camera: [34, ]
        :param train_cameras: [#src, 34]
        :return: [#src, ..., 4]; The first 3 channels are unit-length vector of the difference between
        query and target ray directions, the last channel is the inner product of the two directions.
        """
        train_poses = train_cameras[:, -16:].reshape(-1, 4, 4)  # [#src, 4, 4]
        num_views = len(train_poses)
        query_pose = (
            query_camera[-16:].reshape(-1, 4, 4).repeat(num_views, 1, 1)
        )  # [#src, 4, 4]

        if xyz.ndim == 3:
            # here xyz: [#ray, #sample, 3]
            original_shape = xyz.shape[:2]
            xyz = xyz.reshape(-1, 3)
            ray2tar_pose = query_pose[:, :3, 3].unsqueeze(1) - xyz.unsqueeze(
                0
            )  # [#src, 1, 3] - [1, #ray x #sample, 3] -> [#src, #ray x #sample, 3]
            ray2train_pose = train_poses[:, :3, 3].unsqueeze(1) - xyz.unsqueeze(
                0
            )  # [#src, 1, 3] - [1, #ray x #sample, 3] -> [#src, #ray x #sample, 3]
        else:
            raise AttributeError(xyz.shape)
        ray2tar_pose /= torch.norm(ray2tar_pose, dim=-1, keepdim=True) + 1e-6
        ray2train_pose /= torch.norm(ray2train_pose, dim=-1, keepdim=True) + 1e-6
        ray_diff = ray2tar_pose - ray2train_pose  # [#src, #ray x #sample, 3]
        ray_diff_norm = torch.norm(ray_diff, dim=-1, keepdim=True)
        ray_diff_dot = torch.sum(
            ray2tar_pose * ray2train_pose, dim=-1, keepdim=True
        )  # [#src, #ray x #sample, 1]
        ray_diff_direction = ray_diff / torch.clamp(ray_diff_norm, min=1e-6)
        ray_diff = torch.cat(
            [ray_diff_direction, ray_diff_dot], dim=-1
        )  # [#src, #ray x #sample, 4]
        ray_diff = ray_diff.reshape(
            (num_views,) + original_shape + (4,)
        )  # [#src, #ray, #sample, 4]
        return ray_diff

    def compute(
        self,
        *,
        xyz,
        query_camera,
        train_imgs,
        train_cameras,
        featmaps,
        debug_epipolar=False,
        debug_epipolar_infos=None,
        train_invalid_masks=None,
    ):
        """
        :param xyz: [#ray, #samples, 3]
        :param query_camera: [1, 34], 34 = img_size(2) + intrinsics(16) + extrinsics(16)
        :param train_imgs: [1, #src, h, w, 3]
        :param train_cameras: [1, #src, 34]
        :param featmaps: [#src, d, h, w]
        :return: rgb_feat_sampled: [#ray, #samples, 3+n_feat],
                 ray_diff: [#ray, #samples, 4],
                 mask: [#ray, #samples, 1]
        """
        assert (
            (train_imgs.shape[0] == 1)
            and (train_cameras.shape[0] == 1)
            and (query_camera.shape[0] == 1)
        ), "only support batch_size=1 for now"

        train_imgs = train_imgs.squeeze(0)  # [#src, h, w, 3]
        train_cameras = train_cameras.squeeze(0)  # [#src, 34]
        query_camera = query_camera.squeeze(0)  # [34, ]

        train_imgs = train_imgs.permute(0, 3, 1, 2)  # [#src, 3, h, w]

        if train_invalid_masks is not None:
            assert train_invalid_masks.shape[0] == 1, f"{train_invalid_masks.shape}"
            train_invalid_masks = train_invalid_masks[0, ...].permute(
                0, 3, 1, 2
            )  # [#src, 1, h, w], True indicates invalid

        h, w = train_cameras[0][:2]

        # compute the projection of the query points to each reference image
        # - pixel_locations: [#src, #ray, #sample, 2]
        # - mask_in_front: [#src, #ray, #sample]
        pixel_locations, mask_in_front = self.compute_projections(xyz, train_cameras)

        normalized_pixel_locations = self.normalize(
            pixel_locations, h, w
        )  # [#src, #ray, #sample, 2]

        if debug_epipolar:
            # NOTE: DEBUG
            tmp_train_rgbs = (
                train_imgs.permute(0, 2, 3, 1).cpu().numpy() * 255
            ).astype(
                np.uint8
            )  # [#src, h, w, 3]

            if train_invalid_masks is not None:
                tmp_train_invalid_masks = (
                    train_invalid_masks.permute(0, 2, 3, 1)
                    .expand(-1, -1, -1, 3)
                    .cpu()
                    .numpy()
                    * 255
                ).astype(
                    np.uint8
                )  # [#src, h, w, 3]
                tmp_train_invalid_masks[tmp_train_invalid_masks != 0] = 128

            debug_tgt_img = debug_epipolar_infos["tgt_img"]
            debut_tgt_row, debug_tgt_col = debug_epipolar_infos[
                "tgt_pix_coord"
            ]  # [2, ]

            if pixel_locations.ndim == 4:
                # Here [#src, #ray, #sample, 2]
                pix_coords = pixel_locations.cpu().numpy()[
                    :, 0, ...
                ]  # [#src, #samples, 2]
                tmp_n_views, tmp_n_samples, _ = pix_coords.shape
                tmp_vert_rgbs = debug_epipolar_infos["vert_rgbs"]

                tmp_tgt_img_with_dot = copy.deepcopy(debug_tgt_img)
                tmp_tgt_img_with_dot[
                    (debut_tgt_row - 2) : (debut_tgt_row + 2),
                    (debug_tgt_col - 2) : (debug_tgt_col + 2),
                    :,
                ] = np.array(
                    [[255, 255, 0]], dtype=np.uint8
                )  # yellow
                PIL.Image.fromarray(tmp_tgt_img_with_dot).save(
                    debug_epipolar_infos["debug_dir"] / f"epi_tgt_with_dot.png"
                )

                for tmp_i in range(tmp_n_views):
                    tmp_img2ret = self._vis_epipolar_projection(
                        pix_coords[tmp_i, ...],
                        tmp_vert_rgbs,
                        tmp_train_rgbs[tmp_i, ...],
                        h,
                        w,
                    )
                    PIL.Image.fromarray(tmp_img2ret).save(
                        debug_epipolar_infos["debug_dir"] / f"epi_only_{tmp_i:03d}.png"
                    )
                    tmp_cat_img = np.concatenate(
                        [tmp_tgt_img_with_dot, tmp_img2ret], axis=1
                    )
                    PIL.Image.fromarray(tmp_cat_img).save(
                        debug_epipolar_infos["debug_dir"] / f"epi_cat_{tmp_i:03d}.png"
                    )
                    if train_invalid_masks is not None:
                        tmp_mask2ret = self._vis_epipolar_projection(
                            pix_coords[tmp_i, ...],
                            tmp_vert_rgbs,
                            tmp_train_invalid_masks[tmp_i, ...],
                            h,
                            w,
                        )
                        PIL.Image.fromarray(tmp_mask2ret).save(
                            debug_epipolar_infos["debug_dir"]
                            / f"epi_mask_{tmp_i:03d}.png"
                        )
            else:
                raise AttributeError(pixel_locations.shape)

            import sys

            sys.exit(1)

        # rgb sampling
        if normalized_pixel_locations.ndim == 4:
            rgbs_sampled = F.grid_sample(
                train_imgs, normalized_pixel_locations, align_corners=True
            )  # [#src, 3, H, W], [#src, #ray, #samples, 2] -> [#src, 3, #ray, #sample]
            rgb_sampled = rgbs_sampled.permute(2, 3, 0, 1)  # [#ray, #samples, #src, 3]

            # deep feature sampling
            feat_sampled = F.grid_sample(
                featmaps, normalized_pixel_locations, align_corners=True
            )  # [#src, feat_dim, #ray, #sample]
            feat_sampled = feat_sampled.permute(2, 3, 0, 1)  # [#ray, #samples, #src, d]
            rgb_feat_sampled = torch.cat(
                [rgb_sampled, feat_sampled], dim=-1
            )  # [#ray, #samples, #src, d+3]

            if train_invalid_masks is not None:
                invaid_mask_sampled = F.grid_sample(
                    train_invalid_masks, normalized_pixel_locations, align_corners=True
                )  # [#src, 1, H, W], [#src, #ray, #samples, 2] -> [#src, 1, #ray, #sample]
                invaid_mask_sampled = (
                    invaid_mask_sampled.permute(2, 3, 0, 1) > 1e-3
                ).float()  # [#ray, #samples, #src, 1], True indicates invalid
        else:
            raise AttributeError(normalized_pixel_locations.ndim)

        # mask
        inbound = self.inbound(pixel_locations, h, w)  # [#src, #ray, #sample]
        ray_diff = self.compute_angle(
            xyz, query_camera, train_cameras
        )  # [#src, #ray, #sample, 4]

        if ray_diff.ndim == 4:
            ray_diff = ray_diff.permute(
                1, 2, 0, 3
            )  # [#src, #ray, #sample, 4] -> [#ray, #sample, #src, 4]
            mask_inbound = (
                (inbound * mask_in_front).float().permute(1, 2, 0)[..., None]
            )  # [#src, #ray, #sample] -> [#ray, #sample, #src, 1]

            if train_invalid_masks is not None:
                mask = mask_inbound * (1.0 - invaid_mask_sampled)
            else:
                mask = mask_inbound
        else:
            raise AttributeError(ray_diff.ndim)

        extra_outs = None

        ret_dict = {
            "rgb_feat": rgb_feat_sampled,
            "ray_diff": ray_diff,
            "mask_inbound": mask_inbound,
            "mask": mask,
            "extra": extra_outs,
        }
        if train_invalid_masks is not None:
            ret_dict["mask_invalid"] = invaid_mask_sampled

        return ret_dict

    def _vis_epipolar_projection(self, pix_coords, vert_rgbs, img, h, w):
        row_valid = (pix_coords[:, 1] >= 0) & (pix_coords[:, 1] <= int(h) - 1)
        col_valid = (pix_coords[:, 0] >= 0) & (pix_coords[:, 0] <= int(w) - 1)

        tmp_pix_coords = pix_coords[row_valid & col_valid, :].astype(int)
        tmp_vert_rgbs = vert_rgbs[row_valid & col_valid, :]

        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        tmp_vert_rgb2bgr = tmp_vert_rgbs[:, [2, 1, 0]].tolist()

        for i in range(tmp_vert_rgbs.shape[0]):
            cv2.circle(
                img_bgr,
                tmp_pix_coords[i, :],
                radius=2,
                color=tmp_vert_rgb2bgr[i],
                thickness=-1,
            )

        img2ret = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img2ret
