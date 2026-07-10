"""
Configuration for the SAFEDAM debug tracking wrapper.

This dataclass centralizes the tuning knobs that were previously scattered as
literals / env reads inside ``DAM4SAMMOT.__init__``. It is intentionally
behavior-preserving:

- ``from_env()`` reproduces the exact environment-variable names, defaults and
  parsing (bool/float/int) that the constructor used before.
- Every default here matches the previous hardcoded / getattr-fallback value.

The tracker copies these values onto ``self`` in ``__init__`` so that all
existing read sites (``self.xxx`` and ``getattr(self, "xxx", default)``) keep
working unchanged, including the ability to override an attribute at runtime
after construction.
"""
from dataclasses import dataclass
from typing import Optional


def _env_bool(name, default=False):
    import os
    return os.environ.get(name, "1" if default else "0").lower() in {
        "1", "true", "yes", "on"
    }


@dataclass
class SafeDAMConfig:
    # === Model / image ===
    input_image_size: int = 1008
    fill_hole_area: int = 8

    # === Virtual tracker (env-sourced) ===
    enable_virtual_tracker: bool = True
    sam3_det_thresh: float = 0.8
    vt_alt_bbox_candidate_th: float = 0.7
    vt_alt_bbox_alt_th: float = 0.1
    vt_tracker_overlap_th: float = 0.3
    vt_precheck_overlap_th: float = 0.3
    vt_alt_min_pixels: int = 1
    vt_empty_mask_patience: int = 30
    max_tracker_pool: int = 10

    # === Memory management ===
    max_batch_sz: int = 200
    update_delta: int = 5
    bootstrap_mem_target: int = 6
    max_ram: int = 3
    max_drm: int = 3
    use_last: bool = True
    non_overlap_masks_for_mem_enc: bool = False
    binarize_mask_from_pts_for_mem_enc: bool = True

    # === Adaptive update gate ===
    use_adaptive_update_gate: bool = True
    update_gate_window: int = 20
    update_gate_moving_window: int = 10
    update_gate_warmup: int = 3
    update_gate_ema_alpha: float = 0.1
    update_gate_score_iou_floor: float = 0.0
    update_gate_score_iou_cap: float = 6.0
    update_gate_ref_ratio: float = 0.35
    update_gate_last_ratio: float = 0.0
    update_gate_stats_source: str = "all"
    enable_high_iou_rescue: bool = False
    update_gate_high_iou_rescue: float = 0.9
    update_gate_rescue_ratio: float = 0.5
    update_gate_min_pred_iou: float = 0.0
    update_gate_min_obj_score: float = 0.0
    use_update_gate_continuation_rescue: bool = False
    update_gate_continuation_max_gap: int = 1
    update_gate_continuation_score_iou_floor: float = 1.0
    update_gate_continuation_min_pred_iou: float = 0.65
    update_gate_continuation_min_obj_score: float = 1.4
    update_gate_max_history: int = 300

    # === Overlap resolve / freeze (were getattr-fallback only) ===
    memory_lock_overlap_th: Optional[float] = None  # None -> falls back to vt_tracker_overlap_th
    memory_lock_alt_pred_iou_th: float = 0.7
    overlap_update_freeze_high_th: float = 0.9
    overlap_update_freeze_low_th: float = 0.2

    def __post_init__(self):
        # Original code used getattr(self, "memory_lock_overlap_th",
        # self.vt_tracker_overlap_th): with no explicit value it fell back to
        # vt_tracker_overlap_th. Preserve that by resolving None here.
        if self.memory_lock_overlap_th is None:
            self.memory_lock_overlap_th = self.vt_tracker_overlap_th

    @classmethod
    def from_env(cls):
        """Build config reproducing the previous __init__ env reads exactly."""
        import os
        return cls(
            enable_virtual_tracker=_env_bool("SAFE_DAM_ENABLE_VIRTUAL_TRACKER", True),
            sam3_det_thresh=float(os.environ.get("SAFE_DAM_SAM3_DET_THRESH", "0.8")),
            vt_alt_bbox_candidate_th=float(
                os.environ.get("SAFE_DAM_VT_ALT_BBOX_CANDIDATE_TH", "0.7")
            ),
            vt_alt_bbox_alt_th=float(
                os.environ.get("SAFE_DAM_VT_ALT_BBOX_ALT_TH", "0.1")
            ),
            vt_tracker_overlap_th=float(
                os.environ.get("SAFE_DAM_VT_TRACKER_OVERLAP_TH", "0.3")
            ),
            vt_precheck_overlap_th=float(
                os.environ.get("SAFE_DAM_VT_PRECHECK_OVERLAP_TH", "0.3")
            ),
            vt_alt_min_pixels=int(
                os.environ.get("SAFE_DAM_VT_ALT_MIN_PIXELS", "1")
            ),
            vt_empty_mask_patience=int(
                os.environ.get("SAFE_DAM_VT_EMPTY_MASK_PATIENCE", "30")
            ),
            max_tracker_pool=max(
                1, int(os.environ.get("SAFE_DAM_MAX_TRACKER_POOL", "10"))
            ),
            update_gate_stats_source=os.environ.get(
                "SAFE_DAM_UPDATE_GATE_STATS_SOURCE", "all"
            ).lower(),
        )
