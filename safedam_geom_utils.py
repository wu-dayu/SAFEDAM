"""
Stateless geometry / tensor helpers for the SAFEDAM tracking wrappers.

These functions were extracted verbatim from ``tracking_wrapper_mot_VT_debug.py``
during Phase 2 of the refactor. They hold no tracker state and only operate on
their arguments, so they are safe to share across wrappers.
"""
import numpy as np
import cv2
import torch


def keep_largest_component(mask):
    """
    Keeps only the largest connected component from a binary mask.

    Args:
    - mask (numpy array): 2D binary mask where object pixels are non-zero and background is 0.

    Returns:
    - filtered_mask (numpy array): Binary mask with only the largest connected component.
    """
    # Perform connected components analysis
    _, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    # Find the index of the largest component (excluding background)
    largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])  # Skip background (index 0)
    # Create a mask that contains only the largest component
    filtered_mask = np.zeros_like(mask)
    filtered_mask[labels == largest_component] = 1
    return filtered_mask


def npmask2box(mask):
    # mask is a 2D numpy array in a np.uint8 format
    x_ = np.where(mask.sum(0) > 0)[0]
    y_ = np.where(mask.sum(1) > 0)[0]
    x0, x1 = x_.min(), x_.max()
    y0, y1 = y_.min(), y_.max()
    # convert to (x, y0, width, height) bbox format
    return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]


def largest_component_overlap_ratios(mask_a, mask_b):
    mask_a = np.asarray(mask_a).astype(np.uint8)
    mask_b = np.asarray(mask_b).astype(np.uint8)
    if mask_a.shape != mask_b.shape or mask_a.sum() == 0 or mask_b.sum() == 0:
        return 0.0, 0.0, 0

    mask_a_largest = keep_largest_component(mask_a)
    mask_b_largest = keep_largest_component(mask_b)
    area_a = int(mask_a_largest.sum())
    area_b = int(mask_b_largest.sum())
    if area_a == 0 or area_b == 0:
        return 0.0, 0.0, 0

    intersection = int(np.logical_and(mask_a_largest, mask_b_largest).sum())
    if intersection == 0:
        return 0.0, 0.0, 0
    return intersection / area_a, intersection / area_b, intersection


def bbox_overlap_ratios(bbox_a, bbox_b):
    ax, ay, aw, ah = bbox_a
    bx, by, bw, bh = bbox_b
    area_a = max(0, aw) * max(0, ah)
    area_b = max(0, bw) * max(0, bh)
    if area_a == 0 or area_b == 0:
        return 0.0, 0.0, 0

    inter_x0 = max(ax, bx)
    inter_y0 = max(ay, by)
    inter_x1 = min(ax + aw, bx + bw)
    inter_y1 = min(ay + ah, by + bh)
    inter_w = max(0, inter_x1 - inter_x0)
    inter_h = max(0, inter_y1 - inter_y0)
    intersection = inter_w * inter_h
    if intersection == 0:
        return 0.0, 0.0, 0
    return intersection / area_a, intersection / area_b, intersection


def mask_to_lowres_tensor(mask, target_hw, device):
    mask_tensor = torch.as_tensor(mask, dtype=torch.float32, device=device)[None, None]
    if tuple(mask_tensor.shape[-2:]) != tuple(target_hw):
        mask_tensor = torch.nn.functional.interpolate(
            mask_tensor,
            size=target_hw,
            mode="bilinear",
            align_corners=False,
        )
    return (mask_tensor >= 0.5).float()
