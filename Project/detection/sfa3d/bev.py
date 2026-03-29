from __future__ import annotations

from typing import Optional

import numpy as np
from PIL import Image

from .config import BEVConfig


def sort_and_map(
    pcl: np.ndarray,
    channel_index: int,
    return_counts: bool = False,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    """Sort by x, y, and descending channel value, then keep one point per cell."""

    idx = np.lexsort((-pcl[:, channel_index], pcl[:, 1], pcl[:, 0]))
    pcl_sorted = pcl[idx]
    counts: Optional[np.ndarray] = None

    if return_counts:
        _, indices, counts = np.unique(
            pcl_sorted[:, 0:2],
            axis=0,
            return_index=True,
            return_counts=True,
        )
    else:
        _, indices = np.unique(pcl_sorted[:, 0:2], axis=0, return_index=True)

    return pcl_sorted[indices], counts


def pcl_to_bev(pcl: np.ndarray, config: BEVConfig) -> np.ndarray:
    """Convert a point cloud into a 3-channel BEV map.

    The returned array has shape ``(3, bev_height, bev_width)`` with channels:
    ``BEV[0] -> intensity``, ``BEV[1] -> height``, ``BEV[2] -> density``.
    """

    if pcl.ndim != 2:
        raise ValueError("Point cloud must be a 2D array of shape [n_points, n_channels].")
    if pcl.shape[1] < 4:
        raise ValueError("Point cloud must contain at least x, y, z, intensity channels.")

    pcl = pcl.copy()
    limits = config.limits

    mask = np.where(
        (pcl[:, 0] >= limits.x[0])
        & (pcl[:, 0] <= limits.x[1])
        & (pcl[:, 1] >= limits.y[0])
        & (pcl[:, 1] <= limits.y[1])
        & (pcl[:, 2] >= limits.z[0])
        & (pcl[:, 2] <= limits.z[1])
    )
    pcl = pcl[mask]

    bev_map = np.zeros((3, config.bev_height, config.bev_width), dtype=np.float32)
    if pcl.size == 0:
        return bev_map

    pcl[:, 2] = pcl[:, 2] - limits.z[0]
    pcl[pcl[:, 3] < limits.intensity[0], 3] = limits.intensity[0]
    pcl[pcl[:, 3] > limits.intensity[1], 3] = limits.intensity[1]

    bev_x_discret = (limits.x[1] - limits.x[0]) / config.bev_height
    bev_y_discret = (limits.y[1] - limits.y[0]) / config.bev_width

    pcl[:, 0] = np.int_(np.floor(pcl[:, 0] / bev_x_discret))
    pcl[:, 1] = np.int_(np.floor(pcl[:, 1] / bev_y_discret) + (config.bev_width + 1) / 2)

    pcl_height_sorted, counts = sort_and_map(pcl, 2, return_counts=True)
    xs = np.int_(pcl_height_sorted[:, 0])
    ys = np.int_(pcl_height_sorted[:, 1])

    normalized_height = pcl_height_sorted[:, 2] / float(abs(limits.z[1] - limits.z[0]))
    height_map = np.zeros((config.bev_height + 1, config.bev_width + 1), dtype=np.float32)
    height_map[xs, ys] = normalized_height

    normalized_density = np.minimum(1.0, np.log(counts + 1) / np.log(64))
    density_map = np.zeros((config.bev_height + 1, config.bev_width + 1), dtype=np.float32)
    density_map[xs, ys] = normalized_density

    pcl_int_sorted, _ = sort_and_map(pcl, 3, return_counts=False)
    xs = np.int_(pcl_int_sorted[:, 0])
    ys = np.int_(pcl_int_sorted[:, 1])

    intensity_denom = np.amax(pcl_int_sorted[:, 3]) - np.amin(pcl_int_sorted[:, 3])
    if intensity_denom == 0:
        normalized_int = np.zeros_like(pcl_int_sorted[:, 3], dtype=np.float32)
    else:
        normalized_int = pcl_int_sorted[:, 3] / intensity_denom

    intensity_map = np.zeros((config.bev_height + 1, config.bev_width + 1), dtype=np.float32)
    intensity_map[xs, ys] = normalized_int

    bev_map[2, :, :] = density_map[: config.bev_height, : config.bev_width]
    bev_map[1, :, :] = height_map[: config.bev_height, : config.bev_width]
    bev_map[0, :, :] = intensity_map[: config.bev_height, : config.bev_width]

    return bev_map


def show_bev_map(bev_map: np.ndarray) -> None:
    """Display the BEV map as RGB and per-channel debug images."""

    bev_image = (np.swapaxes(np.swapaxes(bev_map, 0, 1), 1, 2) * 255).astype(np.uint8)
    mask = np.zeros_like(bev_image[:, :, 0])

    height_image = Image.fromarray(np.dstack((bev_image[:, :, 0], mask, mask)))
    den_image = Image.fromarray(np.dstack((mask, bev_image[:, :, 1], mask)))
    int_image = Image.fromarray(np.dstack((mask, mask, bev_image[:, :, 2])))

    int_image.show()
    den_image.show()
    height_image.show()
    Image.fromarray(bev_image).show()
