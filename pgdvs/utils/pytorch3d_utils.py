import torch
import pytorch3d


def cameras_from_opencv_to_pytorch3d(
    R: torch.Tensor,
    tvec: torch.Tensor,
    camera_matrix: torch.Tensor,
    image_size: torch.Tensor,
):
    # Ref: https://github.com/facebookresearch/pytorch3d/blob/57f6e79280e78b6e8308f750e64d32984ddeaba4/pytorch3d/renderer/camera_conversions.py#L19

    focal_length = torch.stack([camera_matrix[:, 0, 0], camera_matrix[:, 1, 1]], dim=-1)
    principal_point = camera_matrix[:, :2, 2]

    # Retype the image_size correctly and flip to width, height.
    image_size_wh = image_size.to(R).flip(dims=(1,))

    # Screen to NDC conversion:
    # For non square images, we scale the points such that smallest side
    # has range [-1, 1] and the largest side has range [-u, u], with u > 1.
    # This convention is consistent with the PyTorch3D renderer, as well as
    # the transformation function `get_ndc_to_screen_transform`.
    scale = image_size_wh.to(R).min(dim=1, keepdim=True)[0] / 2.0
    scale = scale.expand(-1, 2)
    c0 = image_size_wh / 2.0

    # Get the PyTorch3D focal length and principal point.
    focal_pytorch3d = focal_length / scale
    p0_pytorch3d = -(principal_point - c0) / scale

    # For R, T we flip x, y axes (opencv screen space has an opposite
    # orientation of screen axes).
    # We also transpose R (opencv multiplies points from the opposite=left side).
    R_pytorch3d = R.clone().permute(0, 2, 1)  # PyTorch3D is row-major
    T_pytorch3d = tvec.clone()
    R_pytorch3d[:, :, :2] *= -1
    T_pytorch3d[:, :2] *= -1

    return pytorch3d.renderer.PerspectiveCameras(
        R=R_pytorch3d,
        T=T_pytorch3d,
        focal_length=focal_pytorch3d,
        principal_point=p0_pytorch3d,
        image_size=image_size,
        device=R.device,
    )


class SimpleShader(torch.nn.Module):
    # - https://github.com/facebookresearch/pytorch3d/issues/84#issuecomment-590118666
    # - https://github.com/facebookresearch/pytorch3d/issues/607#issuecomment-801241305
    def __init__(self, device="cpu", blend_params=None):
        super().__init__()
        self.blend_params = (
            blend_params
            if blend_params is not None
            else pytorch3d.renderer.BlendParams(background_color=(0.0, 0.0, 0.0))
        )

    def forward(self, fragments, meshes, **kwargs) -> torch.Tensor:
        blend_params = kwargs.get("blend_params", self.blend_params)
        pixel_colors = meshes.sample_textures(fragments)
        images = pytorch3d.renderer.blending.hard_rgb_blend(
            pixel_colors, fragments, blend_params
        )
        return images  # (N, H, W, 3) RGBA image
