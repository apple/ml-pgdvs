import logging
import pathlib

import torch

import pgdvs.utils.softsplat as softsplat


DEBUG_DIR = pathlib.Path(__file__).absolute().parent.parent.parent / "debug"
DEBUG_DIR.mkdir(parents=True, exist_ok=True)


LOGGER = logging.getLogger(__name__)


class PGDVSBaseRenderer(torch.nn.Module):
    def get_batched_rays(
        self, *, device, batch_size, H, W, render_stride, intrinsics, c2w
    ):
        """
        :param H: image height
        :param W: image width
        :param intrinsics: 4 by 4 intrinsic matrix
        :param c2w: 4 by 4 camera to world extrinsic matrix
        :return:
        """
        u, v = torch.meshgrid(
            torch.arange(W, device=device)[::render_stride],
            torch.arange(H, device=device)[::render_stride],
            indexing="xy",
        )  # both are [H, W]

        render_h, render_w = u.shape

        u = u.reshape(-1).float()  # + 0.5    # add half pixel
        v = v.reshape(-1).float()  # + 0.5
        pixels = torch.stack((u, v, torch.ones_like(u)), dim=0)  # (3, H*W)
        batched_pixels = pixels.unsqueeze(0).repeat(batch_size, 1, 1)  # [B, 3, HxW]

        rays_d = (
            c2w[:, :3, :3].bmm(torch.inverse(intrinsics[:, :3, :3])).bmm(batched_pixels)
        ).transpose(
            1, 2
        )  # [B, 3, 3] x [B, 3, 3] x [B, 3, HxW] -> [B, 3, HxW] -> [B, HxW, 3]
        rays_o = c2w[:, :3, 3].unsqueeze(1).repeat(1, rays_d.shape[1], 1)  # [B, HxW, 3]
        rays_d = rays_d.reshape(-1, 3)  # [BxHxW, 3]
        rays_o = rays_o.reshape(-1, 3)  # [BxHxW, 3]
        uvs = batched_pixels[:, :2, :].permute(0, 2, 1).reshape((-1, 2))

        batch_refs = (
            torch.arange(batch_size)
            .reshape((batch_size, 1))
            .expand(-1, u.shape[0])
            .reshape(-1)
        )  # [BxHxW]

        return rays_o, rays_d, uvs, batch_refs, (render_h, render_w)

    def softsplat_img(
        self,
        *,
        rgb_src1,
        flow_src1_to_tgt,
        rgb_src2=None,
        flow_src1_to_src2=None,
        softsplat_metric_src1_to_src2=None,
    ):
        if softsplat_metric_src1_to_src2 is None:
            backwarp_img_for_softsplt_metric = self.backwarp_for_softsplat_metric(
                tenIn=rgb_src2, tenFlow=flow_src1_to_src2
            )  # [B, 3, H, W]
            softsplat_metric_src1_to_src2 = torch.nn.functional.l1_loss(
                input=rgb_src1,
                target=backwarp_img_for_softsplt_metric,
                reduction="none",
            ).mean(
                dim=1, keepdim=True
            )  # [B, 1, H, W]

        splat_img_src1_to_tgt = softsplat.softsplat(
            tenIn=rgb_src1,
            tenFlow=flow_src1_to_tgt,
            tenMetric=(
                -self.softsplat_metric_abs_alpha * softsplat_metric_src1_to_src2
            ).clip(-self.softsplat_metric_abs_alpha, self.softsplat_metric_abs_alpha),
            strMode="soft",
        )  # [B, 3, H, W]

        return splat_img_src1_to_tgt, softsplat_metric_src1_to_src2

    def backwarp_for_softsplat_metric(self, tenIn, tenFlow):
        if not hasattr(self, "backwarp_grid_dict"):
            self.backwarp_grid_dict = {}

        if str(tenFlow.shape) not in self.backwarp_grid_dict:
            tenHor = (
                torch.linspace(
                    start=-1.0,
                    end=1.0,
                    steps=tenFlow.shape[3],
                    dtype=tenFlow.dtype,
                    device=tenFlow.device,
                )
                .view(1, 1, 1, -1)
                .repeat(1, 1, tenFlow.shape[2], 1)
            )
            tenVer = (
                torch.linspace(
                    start=-1.0,
                    end=1.0,
                    steps=tenFlow.shape[2],
                    dtype=tenFlow.dtype,
                    device=tenFlow.device,
                )
                .view(1, 1, -1, 1)
                .repeat(1, 1, 1, tenFlow.shape[3])
            )
            self.backwarp_grid_dict[str(tenFlow.shape)] = torch.cat(
                [tenHor, tenVer], 1
            ).to(tenIn.device)

        tenFlow = torch.cat(
            [
                tenFlow[:, 0:1, :, :] / ((tenIn.shape[3] - 1.0) / 2.0),
                tenFlow[:, 1:2, :, :] / ((tenIn.shape[2] - 1.0) / 2.0),
            ],
            1,
        )

        return torch.nn.functional.grid_sample(
            input=tenIn,
            grid=(self.backwarp_grid_dict[str(tenFlow.shape)] + tenFlow).permute(
                0, 2, 3, 1
            ),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )
