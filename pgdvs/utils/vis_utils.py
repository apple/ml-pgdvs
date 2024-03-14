import os
import cv2
import trimesh
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

import torch

HUGE_NUMBER = 1e10
TINY_NUMBER = 1e-6  # float32 only has 7 decimal digits precision


def get_vertical_colorbar(h, vmin, vmax, cmap_name="jet", label=None, cbar_precision=2):
    """
    :param w: pixels
    :param h: pixels
    :param vmin: min value
    :param vmax: max value
    :param cmap_name:
    :param label
    :return:
    """
    fig = Figure(figsize=(2, 8), dpi=100)
    fig.subplots_adjust(right=1.5)
    canvas = FigureCanvasAgg(fig)

    # Do some plotting.
    ax = fig.add_subplot(111)
    cmap = cm.get_cmap(cmap_name)
    norm = matplotlib.colors.Normalize(vmin=vmin, vmax=vmax)

    tick_cnt = 6
    tick_loc = np.linspace(vmin, vmax, tick_cnt)
    cb1 = matplotlib.colorbar.ColorbarBase(
        ax, cmap=cmap, norm=norm, ticks=tick_loc, orientation="vertical"
    )

    tick_label = [str(np.round(x, cbar_precision)) for x in tick_loc]
    if cbar_precision == 0:
        tick_label = [x[:-2] for x in tick_label]

    cb1.set_ticklabels(tick_label)

    cb1.ax.tick_params(labelsize=18, rotation=0)

    if label is not None:
        cb1.set_label(label)

    fig.tight_layout()

    canvas.draw()
    s, (width, height) = canvas.print_to_buffer()

    im = np.frombuffer(s, np.uint8).reshape((height, width, 4))

    im = im[:, :, :3].astype(np.float32) / 255.0
    if h != im.shape[0]:
        w = int(im.shape[1] / im.shape[0] * h)
        im = cv2.resize(im, (w, h), interpolation=cv2.INTER_AREA)

    return im


def colorize_np(
    x,
    cmap_name="jet",
    mask=None,
    range=None,
    append_cbar=False,
    cbar_in_image=False,
    cbar_precision=2,
):
    """
    turn a grayscale image into a color image
    :param x: input grayscale, [H, W]
    :param cmap_name: the colorization method
    :param mask: the mask image, [H, W]
    :param range: the range for scaling, automatic if None, [min, max]
    :param append_cbar: if append the color bar
    :param cbar_in_image: put the color bar inside the image to keep the output image the same size as the input image
    :return: colorized image, [H, W]
    """
    if range is not None:
        vmin, vmax = range
    elif mask is not None:
        # vmin, vmax = np.percentile(x[mask], (2, 100))
        vmin = np.min(x[mask][np.nonzero(x[mask])])
        vmax = np.max(x[mask])
        # vmin = vmin - np.abs(vmin) * 0.01
        x[np.logical_not(mask)] = vmin
        # print(vmin, vmax)
    else:
        vmin, vmax = np.percentile(x, (1, 100))
        vmax += TINY_NUMBER

    x = np.clip(x, vmin, vmax)
    x = (x - vmin) / (vmax - vmin)
    # x = np.clip(x, 0., 1.)

    cmap = cm.get_cmap(cmap_name)
    x_new = cmap(x)[:, :, :3]

    if mask is not None:
        mask = np.float32(mask[:, :, np.newaxis])
        x_new = x_new * mask + np.ones_like(x_new) * (1.0 - mask)

    cbar = get_vertical_colorbar(
        h=x.shape[0],
        vmin=vmin,
        vmax=vmax,
        cmap_name=cmap_name,
        cbar_precision=cbar_precision,
    )

    if append_cbar:
        if cbar_in_image:
            x_new[:, -cbar.shape[1] :, :] = cbar
        else:
            x_new = np.concatenate(
                (x_new, np.zeros_like(x_new[:, :5, :]), cbar), axis=1
            )
        return x_new
    else:
        return x_new


# tensor
def colorize(
    x, cmap_name="jet", mask=None, range=None, append_cbar=False, cbar_in_image=False
):
    device = x.device
    x = x.cpu().numpy()
    if mask is not None:
        mask = mask.cpu().numpy() > 0.99
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.erode(mask.astype(np.uint8), kernel, iterations=1).astype(bool)

    x = colorize_np(x, cmap_name, mask, range, append_cbar, cbar_in_image)
    x = torch.from_numpy(x).to(device)
    return x


class MplColorHelper:
    def __init__(self, cmap_name, start_val, stop_val):
        self.cmap_name = cmap_name
        self.cmap = plt.get_cmap(cmap_name)
        self.norm = matplotlib.colors.Normalize(vmin=start_val, vmax=stop_val)
        self.scalarMap = cm.ScalarMappable(norm=self.norm, cmap=self.cmap)

    def get_rgb(self, val):
        return self.scalarMap.to_rgba(val)


def hex2rgb(hex_val):
    hex_val = hex_val.lstrip("#")
    rgb = tuple(int(hex_val[i : i + 2], 16) for i in (0, 2, 4))
    return np.array(rgb)


COLOR_GRADS = [
    "YlOrBr",
    "Blues",
    "Purples",
    "Greens",
    "YlOrBr",
    "Reds",
    "Oranges",
    "YlOrRd",
    "OrRd",
    "PuRd",
    "RdPu",
    "BuPu",
    "GnBu",
    "PuBu",
    "YlGnBu",
    "PuBuGn",
    "BuGn",
    "YlGn",
    "Greys",
]


def draw_ray_pcl(idx, px_ray_pts, save_f):
    # px_ray_pts: [N, 3]
    tmp_n = px_ray_pts.shape[0]
    tmp_helper = MplColorHelper(COLOR_GRADS[idx], 0, tmp_n)
    px_ray_pt_rgbs = np.array([tmp_helper.get_rgb(_) for _ in range(tmp_n)])[
        ::-1, :3
    ]  # [N, 3]

    px_ray_pt_rgbs = (px_ray_pt_rgbs * 255).astype(np.uint8)
    px_pcl = trimesh.PointCloud(
        vertices=px_ray_pts, colors=px_ray_pt_rgbs, process=False
    )
    _ = px_pcl.export(save_f)
    return px_ray_pt_rgbs


def interp_head_tail(head_v, tail_v, N=100):
    line_dir = head_v - tail_v
    interp_vs = (
        tail_v[None, :] + np.linspace(0, 1, N).reshape((N, 1)) * line_dir[None, :]
    )  # [N, 3]
    return interp_vs


def draw_cam_mesh(w2c_mat, save_f, tmp_coord=0.1, flag_save=True):
    N = 100

    c2w = np.linalg.inv(w2c_mat)
    c_world = np.matmul(c2w, np.array([0, 0, 0, 1]).reshape((4, 1)))[:3, 0]
    x_world = np.matmul(c2w, np.array([tmp_coord, 0, 0, 1]).reshape((4, 1)))[:3, 0]
    y_world = np.matmul(c2w, np.array([0, tmp_coord, 0, 1]).reshape((4, 1)))[:3, 0]
    z_world = np.matmul(c2w, np.array([0, 0, tmp_coord, 1]).reshape((4, 1)))[:3, 0]

    x_interp_vs = interp_head_tail(x_world, c_world, N=N)
    y_interp_vs = interp_head_tail(y_world, c_world, N=N)
    z_interp_vs = interp_head_tail(z_world, c_world, N=N)

    # RGB for XYZ, [3 * N, 3]
    all_vs = np.concatenate((x_interp_vs, y_interp_vs, z_interp_vs), axis=0)
    all_colors = np.zeros((3 * N, 4), dtype=np.uint8)
    all_colors[:, 3] = 255
    all_colors[:N, 0] = 255  # red, X
    all_colors[N : 2 * N, 1] = 255  # green, Y
    all_colors[2 * N :, 2] = 255  # blue, Z

    if flag_save:
        frame = trimesh.points.PointCloud(vertices=all_vs, colors=all_colors)
        _ = frame.export(save_f)
    else:
        return all_vs, all_colors


def draw_set_poses(poses, save_f, tmp_coord=1.0):
    n = poses.shape[0]
    tmp_placebolder = np.zeros((n, 1, 4))
    tmp_placebolder[:, 0, 3] = 1
    poses = np.concatenate((poses[:, :3, :4], tmp_placebolder), axis=1)

    all_verts = []
    all_colors = []
    for i in range(n):
        tmp_c2w = poses[i, :4, :4]  # [3, 4]
        tmp_verts, tmp_colors = draw_cam_mesh(
            np.linalg.inv(tmp_c2w), None, tmp_coord=tmp_coord, flag_save=False
        )
        tmp_colors = (tmp_colors.astype(float) * (i + 1) / (n + 2)).astype(np.uint8)
        all_verts.append(tmp_verts)
        all_colors.append(tmp_colors)

    all_verts = np.concatenate(all_verts, axis=0)
    all_colors = np.concatenate(all_colors, axis=0)

    frame = trimesh.points.PointCloud(vertices=all_verts, colors=all_colors)
    _ = frame.export(save_f)
