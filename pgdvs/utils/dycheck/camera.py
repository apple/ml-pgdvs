import json
import copy
import PIL.Image
import numpy as np
import os.path as osp
from typing import Optional, Sequence, Tuple, Union, Dict


class DyCheckCamera(object):
    """A generic camera class that potentially distorts rays.

    This camera class uses OpenCV camera model, whhere the local-to-world
    transform assumes (right, down, forward).

    Attributes:
        orientation (np.ndarray): The orientation of the camera of shape (3, 3)
            that maps the world coordinates to local coordinates.
        position (np.ndarray): The position of the camera of shape (3,) in the
            world coordinates.
        focal_length (Union[np.ndarray, float]): The focal length of the camera.
        principal_point (np.ndarray): The principal point of the camera of
            shape (2,)
        image_size (np.ndarray): The image size (W, H).
        skew (Union[np.ndarray, float]): The skewness of the camera.
        pixel_aspect_ratio (Union[np.ndarray, float]): The pixel aspect ratio.
        radial_distortion (Optional[np.ndarray]): The radial distortion of the
            camera of shape (3,).
        tangential_distortion (Optional[np.ndarray]): The tangential distortion
            of the camera of shape (2,).

    Modified from https://github.com/KAIR-BAIR/dycheck/blob/ddf77a4e006fdbc5aed28e0859c216da0de5aff5/dycheck/geometry/camera.py#L245
    """

    def __init__(
        self,
        orientation: np.ndarray,
        position: np.ndarray,
        focal_length: Union[np.ndarray, float],
        principal_point: np.ndarray,
        image_size: np.ndarray,
        skew: Union[np.ndarray, float] = 0.0,
        pixel_aspect_ratio: Union[np.ndarray, float] = 1.0,
        radial_distortion: Optional[np.ndarray] = None,
        tangential_distortion: Optional[np.ndarray] = None,
        *,
        use_center: bool = True,
        use_projective_depth: bool = True,
    ):
        """Constructor for camera class."""
        if radial_distortion is None:
            radial_distortion = np.array([0, 0, 0], np.float32)
        if tangential_distortion is None:
            tangential_distortion = np.array([0, 0], np.float32)

        self.orientation = np.array(orientation, np.float32)
        self.position = np.array(position, np.float32)
        self.focal_length = np.array(focal_length, np.float32)
        self.principal_point = np.array(principal_point, np.float32)
        self.image_size = np.array(image_size, np.uint32)

        # Distortion parameters.
        self.skew = np.array(skew, np.float32)
        self.pixel_aspect_ratio = np.array(pixel_aspect_ratio, np.float32)
        self.radial_distortion = np.array(radial_distortion, np.float32)
        self.tangential_distortion = np.array(tangential_distortion, np.float32)

        self.use_center = use_center
        self.use_projective_depth = use_projective_depth

    @classmethod
    def fromjson(cls, filename):
        with open(filename) as f:
            camera_dict = json.load(f)

        # Fix old camera JSON.
        if "tangential" in camera_dict:
            camera_dict["tangential_distortion"] = camera_dict["tangential"]

        return cls(
            orientation=np.asarray(camera_dict["orientation"]),
            position=np.asarray(camera_dict["position"]),
            focal_length=camera_dict["focal_length"],
            principal_point=np.asarray(camera_dict["principal_point"]),
            image_size=np.asarray(camera_dict["image_size"]),
            skew=camera_dict["skew"],
            pixel_aspect_ratio=camera_dict["pixel_aspect_ratio"],
            radial_distortion=np.asarray(camera_dict["radial_distortion"]),
            tangential_distortion=np.asarray(camera_dict["tangential_distortion"]),
        )

    def rescale_image_domain(self, scale: float) -> "DyCheckCamera":
        """Rescale the image domain of the camera."""
        if scale <= 0:
            raise ValueError("scale needs to be positive.")

        camera = self.copy()
        camera.focal_length *= scale
        camera.principal_point *= scale
        camera.image_size = np.array(
            (
                int(round(self.image_size[0] * scale)),
                int(round(self.image_size[1] * scale)),
            )
        )
        return camera

    def translate(self, transl: np.ndarray) -> "DyCheckCamera":
        """Translate the camera."""
        camera = self.copy()
        camera.position += transl
        return camera

    def rescale(self, scale: float) -> "DyCheckCamera":
        """Rescale the camera."""
        if scale <= 0:
            raise ValueError("scale needs to be positive.")

        camera = self.copy()
        camera.position *= scale
        return camera

    @property
    def has_tangential_distortion(self):
        return any(self.tangential_distortion != 0)

    @property
    def has_radial_distortion(self):
        return any(self.radial_distortion != 0)

    @property
    def distortion(self):
        """Camera distortion parameters compatible with OpenCV.

        Reference:
            https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
        """
        return np.concatenate(
            [
                self.radial_distortion[:2],
                self.tangential_distortion,
                self.radial_distortion[-1:],
            ]
        )

    def copy(self) -> "DyCheckCamera":
        return copy.deepcopy(self)

    @property
    def scale_factor_x(self):
        return self.focal_length

    @property
    def scale_factor_y(self):
        return self.focal_length * self.pixel_aspect_ratio

    @property
    def principal_point_x(self):
        return self.principal_point[0]

    @property
    def principal_point_y(self):
        return self.principal_point[1]

    @property
    def translation(self):
        return -self.orientation @ self.position

    @property
    def optical_axis(self):
        return self.orientation[2, :]

    @property
    def up_axis(self):
        return -self.orientation[1, :]

    @property
    def intrin(self):
        return np.array(
            [
                [self.scale_factor_x, self.skew, self.principal_point_x],
                [0, self.scale_factor_y, self.principal_point_y],
                [0, 0, 1],
            ],
            np.float32,
        )

    @property
    def extrin(self):
        # 4x4 world-to-camera transform.
        return np.concatenate(
            [
                np.concatenate(
                    [self.orientation, self.translation[..., None]], axis=-1
                ),
                np.array([[0, 0, 0, 1]], np.float32),
            ],
            axis=-2,
        )

    def asdict(self):
        return {
            "orientation": self.orientation,
            "position": self.position,
            "focal_length": self.focal_length,
            "principal_point": self.principal_point,
            "image_size": self.image_size,
            "skew": self.skew,
            "pixel_aspect_ratio": self.pixel_aspect_ratio,
            "radial_distortion": self.radial_distortion,
            "tangential_distortion": self.tangential_distortion,
        }
