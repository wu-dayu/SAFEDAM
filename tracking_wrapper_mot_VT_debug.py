"""
safedam tracking wrapper with improved overlap mask selection mechanism.
"""
import os
import numpy as np
import torch

from collections import OrderedDict

from vot.region.raster import calculate_overlaps
from vot.region.shapes import Rectangle

from safedam_geom_utils import (
    keep_largest_component,
    npmask2box,
    largest_component_overlap_ratios,
    bbox_overlap_ratios,
    mask_to_lowres_tensor,
)
from safedam_tracker_config import SafeDAMConfig

from sam3.model.sam3_tracker_utils import fill_holes_in_mask_scores
from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_video_model


def _default_cuda_device(prefer_second=False):
    if not torch.cuda.is_available():
        return "cpu"
    if prefer_second and torch.cuda.device_count() > 1:
        return "cuda:1"
    return "cuda:0"


def _env_device(name, default):
    return torch.device(os.environ.get(name, default))


def build_sam(ckpt_path, device="cuda", mode="eval"):
    sam3_model = build_sam3_video_model(
        checkpoint_path=ckpt_path,
        apply_temporal_disambiguation=False,
        device=device,
    )
    tracker = sam3_model.tracker
    tracker.backbone = sam3_model.detector.backbone
    tracker = tracker.to(device)
    if mode == "eval":
        tracker.eval()
        sam3_model.detector.eval()
    return tracker, sam3_model.detector

def load_confs(chkpt_path, model_size):
    checkpoint = os.path.join(chkpt_path, 'sam3.pt')
    #checkpoint = os.path.join(chkpt_path, 'sam3.1_multiplex.pt')
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"SAM 3 checkpoint not found: {checkpoint}")
        #raise FileNotFoundError(f"SAM 3.1 checkpoint not found: {checkpoint}")
    return checkpoint


class DAM4SAMMOT():
    def __init__(self, model_size='large', checkpoint_dir=None, offload_state_to_cpu=False,
                 config=None):

        # Centralized tuning knobs. from_env() reproduces the previous env reads
        # and defaults exactly; pass an explicit config to override.
        self.config = config if config is not None else SafeDAMConfig.from_env()

        if not checkpoint_dir:
            checkpoint_dir = './checkpoints'
        checkpoint = load_confs(checkpoint_dir, model_size)
        self.device = _env_device("D4SM3_DEVICE", _default_cuda_device(prefer_second=True))
        self.sam, self.detector = build_sam(checkpoint, device=str(self.device))

        self.input_image_size = self.config.input_image_size
        self.fill_hole_area = self.config.fill_hole_area
        self.enable_virtual_tracker = self.config.enable_virtual_tracker
        self.sam3_det_thresh = self.config.sam3_det_thresh
        self.vt_alt_bbox_candidate_th = self.config.vt_alt_bbox_candidate_th
        self.vt_alt_bbox_alt_th = self.config.vt_alt_bbox_alt_th
        self.vt_tracker_overlap_th = self.config.vt_tracker_overlap_th
        self.vt_precheck_overlap_th = self.config.vt_precheck_overlap_th
        self.vt_alt_min_pixels = self.config.vt_alt_min_pixels
        self.vt_empty_mask_patience = self.config.vt_empty_mask_patience
        self.max_tracker_pool = self.config.max_tracker_pool
        if self.enable_virtual_tracker:
            self.detector_processor = Sam3Processor(
                self.detector,
                resolution=self.input_image_size,
                device=str(self.device),
                confidence_threshold=self.sam3_det_thresh,
            )
        else:
            self.detector_processor = None
            self.detector = None
        
        self._img_mean = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)[:, None, None]
        self._img_std = torch.tensor([0.5, 0.5, 0.5], dtype=torch.float32)[:, None, None]
        #self._img_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
        #self._img_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]

        self.img_width = None
        self.img_height = None

        self.frame_index = 0
        self.n_frames = None

        self.maskmem_pos_enc = None
        
        self.output_dict = {'cond_frame_outputs': {}, 
                            'non_cond_frame_outputs': {}, 
                            'maskmem_pos_enc': None, 
                            'per_obj_dict': {}}

        self.mask_inputs_per_obj = {}
        self.output_dict_per_obj = {}
        self.temp_output_dict_per_obj = {}
        self.consolidated_frame_inds = {
            "cond_frame_outputs": set(),  # set containing frame indices
            "non_cond_frame_outputs": set(),  # set containing frame indices
        }

        self.obj_id_to_idx = OrderedDict()
        self.obj_idx_to_id = OrderedDict()
        self.obj_ids = []

        if offload_state_to_cpu:
            self.storage_device = torch.device("cpu")
        else:
            self.storage_device = self.device
        
        self.non_overlap_masks_for_mem_enc = self.config.non_overlap_masks_for_mem_enc
        self.binarize_mask_from_pts_for_mem_enc = self.config.binarize_mask_from_pts_for_mem_enc

        # MOT-specific fields
        self.per_object_outputs_all = {}
        self.per_object_obj_ptr = {}  # separate object pointers since they are updated differently
        self.next_obj_id = 1
        self.all_obj_ids = []
        self.real_obj_ids = []
        self.virtual_obj_ids = set()
        self.virtual_obj_owner = {}
        self.virtual_empty_frame_count = {}
        self.max_batch_sz = self.config.max_batch_sz  # how many objects will be processed together (should not impact tracking)
        self.update_delta = self.config.update_delta  # update every delta frames
        self.bootstrap_mem_target = self.config.bootstrap_mem_target  # count RAM+DRM only, exclude frame-0 anchor
        self.max_ram = self.config.max_ram
        self.max_drm = self.config.max_drm
        self.use_last = self.config.use_last  # always use last frame in RAM
        self.add_to_drm_next = {}  # needed for DRM update (to prevent adding twice the same frame to the memory)

        self.use_adaptive_update_gate = self.config.use_adaptive_update_gate
        self.update_gate_window = self.config.update_gate_window
        self.update_gate_moving_window = self.config.update_gate_moving_window
        self.update_gate_warmup = self.config.update_gate_warmup
        self.update_gate_ema_alpha = self.config.update_gate_ema_alpha
        self.update_gate_score_iou_floor = self.config.update_gate_score_iou_floor
        self.update_gate_score_iou_cap = self.config.update_gate_score_iou_cap
        self.update_gate_ref_ratio = self.config.update_gate_ref_ratio
        self.update_gate_last_ratio = self.config.update_gate_last_ratio
        # "accepted" computes moving_avg/EMA only from frames that pass update_gate.
        # Set to "all" to restore the previous behavior: every positive object score.
        self.update_gate_stats_source = self.config.update_gate_stats_source
        self.enable_high_iou_rescue = self.config.enable_high_iou_rescue
        self.update_gate_high_iou_rescue = self.config.update_gate_high_iou_rescue
        self.update_gate_rescue_ratio = self.config.update_gate_rescue_ratio
        self.update_gate_min_pred_iou = self.config.update_gate_min_pred_iou
        self.update_gate_min_obj_score = self.config.update_gate_min_obj_score
        self.use_update_gate_continuation_rescue = self.config.use_update_gate_continuation_rescue
        self.update_gate_continuation_max_gap = self.config.update_gate_continuation_max_gap
        self.update_gate_continuation_score_iou_floor = self.config.update_gate_continuation_score_iou_floor
        self.update_gate_continuation_min_pred_iou = self.config.update_gate_continuation_min_pred_iou
        self.update_gate_continuation_min_obj_score = self.config.update_gate_continuation_min_obj_score
        self.update_gate_max_history = self.config.update_gate_max_history
        # Overlap resolve / freeze thresholds (previously getattr-fallback only).
        self.memory_lock_overlap_th = self.config.memory_lock_overlap_th
        self.memory_lock_alt_pred_iou_th = self.config.memory_lock_alt_pred_iou_th
        self.overlap_update_freeze_high_th = self.config.overlap_update_freeze_high_th
        self.overlap_update_freeze_low_th = self.config.overlap_update_freeze_low_th
        self.score_iou_history = []
        self.accepted_score_iou_history = []
        self.score_iou_ema = []
        self.accepted_score_iou_ema = []
        self.last_update_score_iou = []
        
    def reset_sequence_state(self, seq_name=None, save_vis_dir=None):
        self.seq_name = seq_name
        self.save_vis_dir = save_vis_dir
        # Keep model weights on GPU, but reset all per-sequence runtime state.
        self.img_width = None
        self.img_height = None
        self.frame_index = 0
        self.n_frames = None

        self.maskmem_pos_enc = None
        self.output_dict = {
            'cond_frame_outputs': {},
            'non_cond_frame_outputs': {},
            'maskmem_pos_enc': None,
            'per_obj_dict': {},
        }

        self.mask_inputs_per_obj = {}
        self.output_dict_per_obj = {}
        self.temp_output_dict_per_obj = {}
        self.consolidated_frame_inds = {
            "cond_frame_outputs": set(),
            "non_cond_frame_outputs": set(),
        }

        self.obj_id_to_idx = OrderedDict()
        self.obj_idx_to_id = OrderedDict()
        self.obj_ids = []

        self.per_object_outputs_all = {}
        self.per_object_obj_ptr = {}
        self.next_obj_id = 1
        self.all_obj_ids = []
        self.real_obj_ids = []
        self.virtual_obj_ids = set()
        self.virtual_obj_owner = {}
        self.virtual_empty_frame_count = {}
        self.add_to_drm_next = {}
        self.overlap_update_freeze_state = {}
        self.object_sizes = []
        self.last_added = []
        self.score_iou_history = []
        self.accepted_score_iou_history = []
        self.score_iou_ema = []
        self.accepted_score_iou_ema = []
        self.last_update_score_iou = []

    def _prepare_image(self, image):
        # image is RGB PIL image: values on range [0, 255]
        # normalize values, resize/pad output to (3x1024x1024)
        img = np.array(image.convert("RGB").resize((self.input_image_size, self.input_image_size)))
        img = img / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).float()
        # normalize
        img -= self._img_mean
        img /= self._img_std
        return img.to(self.device)
    
    def _get_features(self, image, num_obj=1):
        # compute backbone features
        backbone_out = self.sam.forward_image(image)
        # vision_features = backbone_out['vision_features']  # (1, 256, 64, 64)
        vision_pos_enc = backbone_out['vision_pos_enc']  # list: [(1, 256, 256, 256), (1, 256, 128, 128), (1, 256, 64, 64)]
        backbone_fpn = backbone_out['backbone_fpn']  # list: [(1, 32, 256, 256), (1, 64, 128, 128), (1, 256, 64, 64)]
        # Note: vision_features is the same as backbone_fpn[-1]

        batch_size = num_obj
        for i, feat in enumerate(backbone_fpn):
            backbone_fpn[i] = feat.expand(batch_size, -1, -1, -1)
        for i, pos in enumerate(vision_pos_enc):
            vision_pos_enc[i] = pos.expand(batch_size, -1, -1, -1)
        
        expanded_backbone_out = {"backbone_fpn": backbone_fpn, "vision_pos_enc": vision_pos_enc}
        features = self.sam._prepare_backbone_features(expanded_backbone_out)
        _, vision_feats, vision_pos_embeds, feat_sizes = features

        # vision_feats: [(65536, 1, 32), (16384, 1, 64), (4096, 1, 256)]
        # vision_pos_embeds: [(65536, 1, 256), (16384, 1, 256), (4096, 1, 256)]
        # feat_sizes: actual values: [(256, 256), (128, 128), (64, 64)]
        return vision_feats, vision_pos_embeds, feat_sizes

    def _get_maskmem_pos_enc(self, batch_size=1):
        expanded_maskmem_pos_enc = [
            x.expand(batch_size, -1, -1, -1) for x in self.maskmem_pos_enc
        ]
        return expanded_maskmem_pos_enc

    def _get_current_update_delta(self, obj_mem):
        return self.update_delta

    def _ensure_update_gate_state(self, obj_idx):
        while len(self.score_iou_history) <= obj_idx:
            self.score_iou_history.append([])
            self.accepted_score_iou_history.append([])
            self.score_iou_ema.append(None)
            self.accepted_score_iou_ema.append(None)
            self.last_update_score_iou.append(None)

    def _recent_median(self, values, window=None):
        if len(values) == 0:
            return None
        if window is None:
            window = self.update_gate_window
        return float(np.median(values[-window:]))

    def _recent_mean(self, values, window=None):
        if len(values) == 0:
            return None
        if window is None:
            window = self.update_gate_moving_window
        return float(np.mean(values[-window:]))

    def _format_gate_value(self, value):
        if value is None:
            return "None"
        return f"{value:.4f}"

    def _get_adaptive_update_gate(
        self,
        obj_idx,
        n_pixels_pos,
        max_pred_iou,
        obj_score,
        score_iou,
        last_ram_frame=None,
    ):
        self._ensure_update_gate_state(obj_idx)

        accepted_scores = self.accepted_score_iou_history[obj_idx]
        all_scores = self.score_iou_history[obj_idx]
        stats_source = getattr(self, "update_gate_stats_source", "accepted")
        if stats_source == "all":
            moving_avg_scores = all_scores
            ema = self.score_iou_ema[obj_idx]
        else:
            moving_avg_scores = accepted_scores
            ema = self.accepted_score_iou_ema[obj_idx]
        accepted_median = self._recent_median(accepted_scores, window=len(accepted_scores))
        moving_avg = self._recent_mean(moving_avg_scores)
        last_score = self.last_update_score_iou[obj_idx]

        threshold = self.update_gate_score_iou_floor
        is_warmup = len(accepted_scores) < self.update_gate_warmup
        if self.use_adaptive_update_gate and not is_warmup:
            refs = [
                value for value in (accepted_median, moving_avg, ema)
                if value is not None
            ]
            if len(refs) > 0:
                threshold = max(threshold, self.update_gate_ref_ratio * min(refs))
            if last_score is not None:
                threshold = max(threshold, self.update_gate_last_ratio * last_score)
            threshold = min(threshold, self.update_gate_score_iou_cap)

        rescue_threshold = max(
            self.update_gate_score_iou_floor,
            self.update_gate_rescue_ratio * threshold,
        )
        visible_gate = n_pixels_pos > 0
        score_gate = score_iou >= threshold
        high_iou_rescue_gate = (
            self.enable_high_iou_rescue
            and max_pred_iou >= self.update_gate_high_iou_rescue
            and score_iou >= rescue_threshold
        )
        ram_gap = None
        if last_ram_frame is not None:
            ram_gap = self.frame_index - last_ram_frame
        continuation_rescue_gate = (
            self.use_update_gate_continuation_rescue
            and ram_gap is not None
            and ram_gap <= self.update_gate_continuation_max_gap
            and score_iou >= self.update_gate_continuation_score_iou_floor
            and max_pred_iou >= self.update_gate_continuation_min_pred_iou
            and obj_score > self.update_gate_continuation_min_obj_score
        )
        update_gate = (
            visible_gate
            and obj_score > self.update_gate_min_obj_score
            and max_pred_iou >= self.update_gate_min_pred_iou
            and (score_gate or high_iou_rescue_gate or continuation_rescue_gate)
        )

        gate_stats = {
            "threshold": threshold,
            "rescue_threshold": rescue_threshold,
            "accepted_median": accepted_median,
            "moving_avg": moving_avg,
            "moving_median": moving_avg,
            "ema": ema,
            "last_score": last_score,
            "score_gate": score_gate,
            "high_iou_rescue_gate": high_iou_rescue_gate,
            "continuation_rescue_gate": continuation_rescue_gate,
            "ram_gap": ram_gap,
            "is_warmup": is_warmup,
        }
        return update_gate, gate_stats

    def _record_score_iou_observation(self, obj_idx, score_iou, obj_score):
        if obj_score <= 0:
            return
        self._ensure_update_gate_state(obj_idx)
        score_iou = float(score_iou)
        self.score_iou_history[obj_idx].append(score_iou)
        if len(self.score_iou_history[obj_idx]) > self.update_gate_max_history:
            self.score_iou_history[obj_idx].pop(0)

        ema = self.score_iou_ema[obj_idx]
        if ema is None:
            self.score_iou_ema[obj_idx] = score_iou
        else:
            alpha = self.update_gate_ema_alpha
            self.score_iou_ema[obj_idx] = alpha * score_iou + (1.0 - alpha) * ema

    def _record_memory_update_trigger(self, obj_idx, score_iou, obj_score):
        if obj_score <= 0:
            return
        self._ensure_update_gate_state(obj_idx)
        score_iou = float(score_iou)
        self.accepted_score_iou_history[obj_idx].append(score_iou)
        if len(self.accepted_score_iou_history[obj_idx]) > self.update_gate_max_history:
            self.accepted_score_iou_history[obj_idx].pop(0)
        ema = self.accepted_score_iou_ema[obj_idx]
        if ema is None:
            self.accepted_score_iou_ema[obj_idx] = score_iou
        else:
            alpha = self.update_gate_ema_alpha
            self.accepted_score_iou_ema[obj_idx] = alpha * score_iou + (1.0 - alpha) * ema
        self.last_update_score_iou[obj_idx] = score_iou

    def _has_active_tracker_overlap_issue(self, masks):
        if len(self.all_obj_ids) <= 1:
            return False
        for obj_idx_i in range(len(self.all_obj_ids)):
            for obj_idx_j in range(obj_idx_i + 1, len(self.all_obj_ids)):
                obj_id_i = self.all_obj_ids[obj_idx_i]
                obj_id_j = self.all_obj_ids[obj_idx_j]
                if not self._objects_overlap_visible(obj_id_i, obj_id_j):
                    continue
                ratio_i, ratio_j, intersection = largest_component_overlap_ratios(
                    masks[obj_idx_i], masks[obj_idx_j]
                )
                if (
                    ratio_i >= self.vt_precheck_overlap_th
                    or ratio_j >= self.vt_precheck_overlap_th
                ):
                    return True
        return False

    def _objects_overlap_visible(self, obj_id_i, obj_id_j):
        i_real = obj_id_i in self.real_obj_ids
        j_real = obj_id_j in self.real_obj_ids
        if i_real and j_real:
            return True
        owner_i = self.virtual_obj_owner.get(obj_id_i)
        owner_j = self.virtual_obj_owner.get(obj_id_j)
        if i_real:
            return owner_j == obj_id_i
        if j_real:
            return owner_i == obj_id_j
        return owner_i is not None and owner_i == owner_j

    def _remove_virtual_trackers(self, obj_ids_to_remove):
        for obj_id in obj_ids_to_remove:
            if obj_id not in self.virtual_obj_ids:
                continue
            obj_idx = self.all_obj_ids.index(obj_id)
            self.all_obj_ids.pop(obj_idx)
            self.virtual_obj_ids.discard(obj_id)
            self.virtual_obj_owner.pop(obj_id, None)
            self.virtual_empty_frame_count.pop(obj_id, None)
            self.per_object_outputs_all.pop(obj_id, None)
            self.per_object_obj_ptr.pop(obj_id, None)
            self.add_to_drm_next.pop(obj_id, None)
            if hasattr(self, "overlap_update_freeze_state"):
                self.overlap_update_freeze_state.pop(obj_id, None)
            if obj_idx < len(self.object_sizes):
                self.object_sizes.pop(obj_idx)
            if obj_idx < len(self.last_added):
                self.last_added.pop(obj_idx)
            if obj_idx < len(self.score_iou_history):
                self.score_iou_history.pop(obj_idx)
            if obj_idx < len(self.accepted_score_iou_history):
                self.accepted_score_iou_history.pop(obj_idx)
            if obj_idx < len(self.score_iou_ema):
                self.score_iou_ema.pop(obj_idx)
            if obj_idx < len(self.accepted_score_iou_ema):
                self.accepted_score_iou_ema.pop(obj_idx)
            if obj_idx < len(self.last_update_score_iou):
                self.last_update_score_iou.pop(obj_idx)

    def _candidate_overlaps_existing_tracker(self, candidate_mask, tracker_masks, tracker_obj_ids=None):
        max_ratio_candidate = 0.0
        max_ratio_tracker = 0.0
        max_tracker_obj_id = None
        for tracker_idx, tracker_mask in enumerate(tracker_masks):
            ratio_candidate, ratio_tracker, _ = largest_component_overlap_ratios(
                candidate_mask, tracker_mask
            )
            if max(ratio_candidate, ratio_tracker) > max(max_ratio_candidate, max_ratio_tracker):
                max_tracker_obj_id = (
                    tracker_obj_ids[tracker_idx]
                    if tracker_obj_ids is not None and tracker_idx < len(tracker_obj_ids)
                    else None
                )
            max_ratio_candidate = max(max_ratio_candidate, ratio_candidate)
            max_ratio_tracker = max(max_ratio_tracker, ratio_tracker)
            if (
                ratio_candidate >= self.vt_tracker_overlap_th
                or ratio_tracker >= self.vt_tracker_overlap_th
            ):
                reject_obj_id = (
                    tracker_obj_ids[tracker_idx]
                    if tracker_obj_ids is not None and tracker_idx < len(tracker_obj_ids)
                    else None
                )
                return True, ratio_candidate, ratio_tracker, reject_obj_id
        return False, max_ratio_candidate, max_ratio_tracker, max_tracker_obj_id

    def _build_vt_alternative_evidence(self, alternative_masks, chosen_mask):
        chosen_mask = np.asarray(chosen_mask).astype(bool)
        evidence_masks = []
        for alt_mask in alternative_masks:
            alt_mask = np.asarray(alt_mask).squeeze().astype(bool)
            alt_mask = np.logical_and(alt_mask, np.logical_not(chosen_mask)).astype(np.uint8)
            if int(alt_mask.sum()) < self.vt_alt_min_pixels:
                continue
            alt_largest = keep_largest_component(alt_mask)
            if int(alt_largest.sum()) < self.vt_alt_min_pixels:
                continue
            evidence_masks.append(alt_largest.astype(bool))
        return evidence_masks

    def _candidate_matches_alternative_evidence(self, candidate_mask, alternative_evidence):
        if not alternative_evidence or not np.any(candidate_mask):
            return False, 0.0, 0.0
        candidate_bbox = npmask2box(candidate_mask)
        best_ratio_candidate = 0.0
        best_ratio_alt = 0.0
        for evidence_mask in alternative_evidence:
            alternative_bbox = npmask2box(evidence_mask)
            ratio_candidate, ratio_alt, _ = bbox_overlap_ratios(
                candidate_bbox, alternative_bbox
            )
            best_ratio_candidate = max(best_ratio_candidate, ratio_candidate)
            best_ratio_alt = max(best_ratio_alt, ratio_alt)
            if (
                ratio_candidate >= self.vt_alt_bbox_candidate_th
                and ratio_alt >= self.vt_alt_bbox_alt_th
            ):
                return True, ratio_candidate, ratio_alt
        return False, best_ratio_candidate, best_ratio_alt

    def _resolve_memory_lock_selection(self, alternative_masks_all, all_ious):
        """
        Multi-round alternative selection for memory lock.

        Returns a per-object list of dicts:
            {
                "candidate_idx": int | None,
                "mask": np.ndarray | None,
                "pred_iou": float,
            }
        """
        num_objs = len(self.all_obj_ids)
        if num_objs == 0:
            return []

        overlap_th = float(getattr(self, "memory_lock_overlap_th", self.vt_tracker_overlap_th))
        alt_pred_iou_th = float(getattr(self, "memory_lock_alt_pred_iou_th", 0.7))

        candidate_orders = []
        candidate_scores = []
        candidate_masks = []
        for obj_idx in range(num_objs):
            scores = np.asarray(all_ious[obj_idx]).astype(float)
            masks = np.asarray(alternative_masks_all[obj_idx])
            chosen_idx = int(np.argmax(scores)) if len(scores) > 0 else None
            valid_indices = [
                int(mask_idx)
                for mask_idx in np.argsort(-scores)
                if (
                    mask_idx < len(masks)
                    and np.asarray(masks[mask_idx]).sum() > 0
                    and (
                        int(mask_idx) == chosen_idx
                        or float(scores[mask_idx]) > alt_pred_iou_th
                    )
                )
            ]
            candidate_orders.append(valid_indices)
            candidate_scores.append(scores)
            candidate_masks.append(masks)

        selected_order_pos = [0 for _ in range(num_objs)]
        selected_candidate_idx = [
            order[0] if len(order) > 0 else None for order in candidate_orders
        ]

        def _current_mask(obj_idx):
            cand_idx = selected_candidate_idx[obj_idx]
            if cand_idx is None:
                return None
            return np.asarray(candidate_masks[obj_idx][cand_idx]).astype(np.uint8)

        while True:
            locked_obj_idxs = set()
            for obj_idx_i in range(num_objs):
                cand_idx_i = selected_candidate_idx[obj_idx_i]
                mask_i = _current_mask(obj_idx_i)
                if cand_idx_i is None or mask_i is None or mask_i.sum() == 0:
                    continue
                for obj_idx_j in range(obj_idx_i + 1, num_objs):
                    cand_idx_j = selected_candidate_idx[obj_idx_j]
                    mask_j = _current_mask(obj_idx_j)
                    if cand_idx_j is None or mask_j is None or mask_j.sum() == 0:
                        continue
                    obj_id_i = self.all_obj_ids[obj_idx_i]
                    obj_id_j = self.all_obj_ids[obj_idx_j]
                    if not self._objects_overlap_visible(obj_id_i, obj_id_j):
                        continue

                    ratio_i, ratio_j, _ = largest_component_overlap_ratios(mask_i, mask_j)
                    if ratio_i < overlap_th and ratio_j < overlap_th:
                        continue

                    score_i = float(candidate_scores[obj_idx_i][cand_idx_i])
                    score_j = float(candidate_scores[obj_idx_j][cand_idx_j])
                    if score_i < score_j or (
                        score_i == score_j and obj_idx_i > obj_idx_j
                    ):
                        locked_obj_idxs.add(obj_idx_i)
                    else:
                        locked_obj_idxs.add(obj_idx_j)

            if not locked_obj_idxs:
                break

            progress = False
            for obj_idx in sorted(locked_obj_idxs):
                next_pos = selected_order_pos[obj_idx] + 1
                selected_order_pos[obj_idx] = next_pos
                if next_pos < len(candidate_orders[obj_idx]):
                    selected_candidate_idx[obj_idx] = candidate_orders[obj_idx][next_pos]
                    progress = True
                else:
                    selected_candidate_idx[obj_idx] = None
                    progress = True

            if not progress:
                break

        resolved = []
        for obj_idx in range(num_objs):
            cand_idx = selected_candidate_idx[obj_idx]
            if cand_idx is None:
                resolved.append({
                    "candidate_idx": None,
                    "mask": None,
                    "pred_iou": 0.0,
                })
                continue
            chosen_idx = int(np.argmax(candidate_scores[obj_idx]))
            if (
                int(cand_idx) != chosen_idx
                and float(candidate_scores[obj_idx][cand_idx]) <= alt_pred_iou_th
            ):
                resolved.append({
                    "candidate_idx": None,
                    "mask": None,
                    "pred_iou": 0.0,
                })
                continue
            resolved.append({
                "candidate_idx": int(cand_idx),
                "mask": np.asarray(candidate_masks[obj_idx][cand_idx]).astype(np.uint8),
                "pred_iou": float(candidate_scores[obj_idx][cand_idx]),
            })
        return resolved

    def _initialize_virtual_tracker(self, mask, img, feats, pos, feat_sizes, owner_obj_id):
        mask_inputs_orig = torch.as_tensor(mask, dtype=torch.float32)[None, None]
        mask_inputs_orig = mask_inputs_orig.to(feats[0].device)
        mask_inputs = torch.nn.functional.interpolate(
            mask_inputs_orig,
            size=(self.sam.image_size, self.sam.image_size),
            align_corners=False,
            mode="bilinear",
            antialias=True,
        )
        mask_inputs = (mask_inputs >= 0.5).float()
        current_out = self.sam.track_step(
            frame_idx=self.frame_index,
            is_init_cond_frame=True,
            current_vision_feats=feats,
            current_vision_pos_embeds=pos,
            feat_sizes=feat_sizes,
            image=img,
            point_inputs=None,
            mask_inputs=mask_inputs,
            output_dict={'per_obj_dict': {}, 'maskmem_pos_enc': None},
            num_frames=self.n_frames,
            track_in_reverse=False,
            run_mem_encoder=False,
            prev_sam_mask_logits=None,
        )
        pred_masks = current_out["pred_masks"]
        if self.fill_hole_area > 0:
            pred_masks = fill_holes_in_mask_scores(pred_masks, self.fill_hole_area)
        pred_masks = pred_masks.to(img.device, non_blocking=True)
        high_res_masks = torch.nn.functional.interpolate(
            pred_masks,
            size=(self.sam.image_size, self.sam.image_size),
            mode="bilinear",
            align_corners=False,
        )
        maskmem_features, _ = self.sam._encode_new_memory(
            image=img,
            current_vision_feats=feats,
            feat_sizes=feat_sizes,
            pred_masks_high_res=high_res_masks,
            object_score_logits=current_out["object_score_logits"],
            is_mask_from_pts=True,
        )
        maskmem_features = maskmem_features.to(torch.bfloat16).to(
            img.device, non_blocking=True
        )

        obj_id = self.next_obj_id
        self.next_obj_id += 1
        self.per_object_outputs_all[obj_id] = [{
            "maskmem_features": maskmem_features,
            "pred_masks": pred_masks,
            "is_init": True,
            "frame_idx": self.frame_index,
            "is_drm": False,
        }]
        self.per_object_obj_ptr[obj_id] = [{
            "obj_ptr": current_out["obj_ptr"],
            "frame_idx": self.frame_index,
            "is_init": True,
        }]
        self.add_to_drm_next[obj_id] = None
        self.all_obj_ids.append(obj_id)
        self.virtual_obj_ids.add(obj_id)
        self.virtual_obj_owner[obj_id] = owner_obj_id
        self.virtual_empty_frame_count[obj_id] = 0
        self.object_sizes.append([])
        self.last_added.append(-1)
        self._ensure_update_gate_state(len(self.all_obj_ids) - 1)

    def _run_virtual_detector(self, image, sources):
        state = self.detector_processor.set_image(image)
        candidates = []

        def append_state_candidates(detector_state, source):
            masks = detector_state.get("masks")
            scores = detector_state.get("scores")
            if masks is None or scores is None:
                return
            masks = masks.detach().cpu().numpy()
            scores = scores.detach().float().cpu().numpy()
            sx, sy = source["center"]
            for mask, score in zip(masks, scores):
                mask = np.asarray(mask).squeeze().astype(bool)
                if not np.any(mask):
                    continue
                ys, xs = np.where(mask)
                center = (float(xs.mean()), float(ys.mean()))
                distance = (center[0] - sx) ** 2 + (center[1] - sy) ** 2
                candidates.append({
                    "mask": mask,
                    "score": float(score),
                    "distance": float(distance),
                    "source_obj_id": source["obj_id"],
                    "source": source,
                })

        for source in sources:
            self.detector_processor.reset_all_prompts(state)
            state = self.detector_processor.add_geometric_prompt(
                source["box"], True, state
            )
            state = self.detector_processor._forward_grounding(state)
            append_state_candidates(state, source)
        return candidates

    def _maybe_spawn_virtual_trackers(self, image, masks, drm_sources, img, feats, pos, feat_sizes):
        if not self.enable_virtual_tracker:
            return
        available_slots = self.max_tracker_pool - len(self.all_obj_ids)
        if available_slots <= 0:
            return

        sources = drm_sources
        if not sources:
            return
        if self._has_active_tracker_overlap_issue(masks):
            return

        candidates = self._run_virtual_detector(image, sources)
        candidates.sort(key=lambda candidate: (candidate["distance"], -candidate["score"]))
        selected_masks_by_owner = {}
        selected = []
        rejected_alt = 0
        rejected_overlap = 0
        for candidate_idx, candidate in enumerate(candidates):
            candidate_mask = candidate["mask"]
            source = candidate["source"]

            alt_passed, best_alt_candidate_ratio, best_alt_ratio = (
                self._candidate_matches_alternative_evidence(
                    candidate_mask, source["alternative_evidence"]
                )
            )
            source_obj_id = candidate["source_obj_id"]
            if not alt_passed:
                rejected_alt += 1
                continue

            visible_covered_masks = []
            visible_covered_obj_ids = []
            for tracker_idx, tracker_obj_id in enumerate(self.all_obj_ids):
                if (
                    tracker_obj_id in self.real_obj_ids
                    or self.virtual_obj_owner.get(tracker_obj_id) == source_obj_id
                ):
                    visible_covered_masks.append(masks[tracker_idx].astype(bool))
                    visible_covered_obj_ids.append(tracker_obj_id)
            for selected_idx, selected_mask in enumerate(selected_masks_by_owner.get(source_obj_id, [])):
                visible_covered_masks.append(selected_mask)
                visible_covered_obj_ids.append(f"selected_{source_obj_id}_{selected_idx}")

            covered, ratio_candidate, ratio_tracker, overlap_obj_id = self._candidate_overlaps_existing_tracker(
                candidate_mask, visible_covered_masks, visible_covered_obj_ids
            )
            if covered:
                rejected_overlap += 1
                continue
            selected.append(candidate)
            selected_masks_by_owner.setdefault(source_obj_id, []).append(candidate_mask)
            if len(selected) >= available_slots:
                break

        for candidate in selected:
            candidate_mask = candidate["mask"]
            candidate_score = candidate["score"]
            candidate_distance = candidate["distance"]
            max_tracker_candidate_ratio = 0.0
            max_tracker_ratio = 0.0
            for tracker_mask in masks:
                ratio_candidate, ratio_tracker, _ = largest_component_overlap_ratios(
                    candidate_mask, tracker_mask
                )
                max_tracker_candidate_ratio = max(max_tracker_candidate_ratio, ratio_candidate)
                max_tracker_ratio = max(max_tracker_ratio, ratio_tracker)
            try:
                if getattr(self, "save_vis_dir", None) and getattr(self, "seq_name", None):
                    import os
                    from PIL import Image
                    vt_init_dir = os.path.join(self.save_vis_dir, self.seq_name, "VT_init_mask")
                    os.makedirs(vt_init_dir, exist_ok=True)
                    
                    img_np = np.array(image.convert("RGB"))
                    mask_colored = np.zeros_like(img_np)
                    mask_colored[:, :, 0] = 255 # red
                    
                    alpha = 0.5
                    overlay = img_np.copy()
                    overlay[candidate_mask] = (overlay[candidate_mask] * (1 - alpha) + mask_colored[candidate_mask] * alpha).astype(np.uint8)
                    
                    filename = (
                        f"{self.seq_name}_frame_{self.frame_index:04d}_score_{candidate_score:.3f}_"
                        f"cand_ratio_{max_tracker_candidate_ratio:.3f}_tracker_ratio_{max_tracker_ratio:.3f}_"
                        f"dist_{candidate_distance:.1f}.jpg"
                    )
                    filepath = os.path.join(vt_init_dir, filename)
                    Image.fromarray(overlay).save(filepath)
            except Exception as e:
                print("Failed to save VT init mask viz:", e)

            self._initialize_virtual_tracker(
                candidate_mask,
                img,
                feats,
                pos,
                feat_sizes,
                owner_obj_id=candidate["source_obj_id"],
            )


    # *****************************************************************
    # **                        VOT Tracker                          **
    # *****************************************************************
    def initialize(self, image, init_regions):
        self.frame_index = 0
        
        if self.img_width is None or self.img_height is None:
            self.img_width = image.width
            self.img_height = image.height

        # prepare image
        img = self._prepare_image(image)
        img = img.unsqueeze(0)  # (1, 3, 1024, 1024)
        
        # compute features
        feats, pos, feat_sizes = self._get_features(img)  # Note: removed number of objects

        self.object_sizes = []
        self.last_added = []

        # take all unmatched detections and put them in memory for future tracking
        for reg in init_regions:
            # support both - bbox and mask initialization
            if 'mask' in reg:
                mask = reg['mask']
                if not isinstance(mask, torch.Tensor):
                    mask = torch.tensor(mask, dtype=torch.bool)
                mask_H, mask_W = mask.shape
                mask_inputs_orig = mask[None, None]  # add batch and channel dimension
                mask_inputs_orig = mask_inputs_orig.float().to(feats[0].device)

                # resize the mask if it doesn't match the model's image size
                if mask_H != self.sam.image_size or mask_W != self.sam.image_size:
                    mask_inputs = torch.nn.functional.interpolate(
                        mask_inputs_orig,
                        size=(self.sam.image_size, self.sam.image_size),
                        align_corners=False,
                        mode="bilinear",
                        antialias=True,  # use antialias for downsampling
                    )
                    mask_inputs_ = (mask_inputs >= 0.5).float()
                else:
                    mask_inputs_ = mask_inputs_orig
                
                point_inputs_ = None

                self.object_sizes.append([])
                self.last_added.append(-1)
                self._ensure_update_gate_state(len(self.object_sizes) - 1)
            elif 'bbox' in reg:
                bbox = reg['bbox']
                box = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]

                points = torch.zeros(0, 2, dtype=torch.float32)
                labels = torch.zeros(0, dtype=torch.int32)
                if points.dim() == 2:
                    points = points.unsqueeze(0)  # add batch dimension
                if labels.dim() == 1:
                    labels = labels.unsqueeze(0)  # add batch dimension
                    
                box = torch.tensor(box, dtype=torch.float32, device=points.device)
                box_coords = box.reshape(1, 2, 2)
                box_labels = torch.tensor([2, 3], dtype=torch.int32, device=labels.device)
                box_labels = box_labels.reshape(1, 2)
                points = torch.cat([box_coords, points], dim=1)
                labels = torch.cat([box_labels, labels], dim=1)
                points = points / torch.tensor([image.width, image.height]).to(points.device)
                
                points = points * self.sam.image_size
                points = points.to(feats[0].device)
                labels = labels.to(feats[0].device)
                
                point_inputs_ = {"point_coords": points, "point_labels": labels}
                mask_inputs_ = None

                self.object_sizes.append([])
                self.last_added.append(-1)
                self._ensure_update_gate_state(len(self.object_sizes) - 1)
            else:
                print('Error: Input region should be mask or rectangle.')
                exit(-1)

            output_dict_ = {'per_obj_dict': {}, 'maskmem_pos_enc': None}
            current_out = self.sam.track_step(
                frame_idx=self.frame_index,
                is_init_cond_frame=True,
                current_vision_feats=feats,
                current_vision_pos_embeds=pos,
                feat_sizes=feat_sizes,
                image=img,
                point_inputs=point_inputs_,
                mask_inputs=mask_inputs_,
                output_dict=output_dict_,
                num_frames=self.n_frames,
                track_in_reverse=False,
                run_mem_encoder=False,  # We might need to put this on True since it is not run separately
                prev_sam_mask_logits=None,
            )          
            pred_masks_gpu = current_out["pred_masks"]

            # potentially fill holes in the predicted masks
            if self.fill_hole_area > 0:
                pred_masks_gpu = fill_holes_in_mask_scores(
                    pred_masks_gpu, self.fill_hole_area
                )
                
            pred_masks = pred_masks_gpu.to(img.device, non_blocking=True)

            high_res_masks = torch.nn.functional.interpolate(
                pred_masks,
                size=(self.sam.image_size, self.sam.image_size),
                mode="bilinear",
                align_corners=False,
            )

            maskmem_features, maskmem_pos_enc = self.sam._encode_new_memory(
                image=img,
                current_vision_feats=feats,
                feat_sizes=feat_sizes,
                pred_masks_high_res=high_res_masks,
                object_score_logits=current_out['object_score_logits'],
                is_mask_from_pts=True
            )

            maskmem_features = maskmem_features.to(torch.bfloat16)
            maskmem_features = maskmem_features.to(img.device, non_blocking=True)
            
            if self.maskmem_pos_enc is None:
                self.maskmem_pos_enc = [x[0:1].clone() for x in maskmem_pos_enc]
                maskmem_pos_enc_ = self.maskmem_pos_enc[0].to(img.device)
                self.output_dict['maskmem_pos_enc'] = maskmem_pos_enc_

            per_obj_dict = {
                "maskmem_features": maskmem_features,  # (1, 64, 64, 64)
                "pred_masks": pred_masks,  # (1, 1, 256, 256)
                "is_init": True, "frame_idx": self.frame_index, "is_drm": False
            }

            # obj_ptr dimmension: (1, 256)
            per_obj_obj_ptr_dict = {"obj_ptr": current_out["obj_ptr"], "frame_idx": self.frame_index, "is_init": True}
            
            self.per_object_outputs_all[self.next_obj_id] = [per_obj_dict]
            self.per_object_obj_ptr[self.next_obj_id] = [per_obj_obj_ptr_dict]
            self.add_to_drm_next[self.next_obj_id] = None
            self.all_obj_ids.append(self.next_obj_id)
            self.real_obj_ids.append(self.next_obj_id)
            self.next_obj_id += 1
        
        return None

    def _run_sam3_inference(self, img, feats, pos, feat_sizes):
        """Run the SAM3 tracker for the current frame across all objects.

        Read-only w.r.t. tracker state: it reads self.per_object_outputs_all,
        self.per_object_obj_ptr, self.output_dict, self.all_obj_ids and
        self.max_batch_sz, then returns a single merged ``current_out`` dict.
        The batch loop only matters when huge numbers of objects are tracked
        (MOT); for VOT (<=10 objects) n_runs is always 1.
        """
        output_dict_ = {
            'per_obj_dict': self.per_object_outputs_all,
            'per_obj_obj_ptr_dict': self.per_object_obj_ptr,
            'maskmem_pos_enc': self.output_dict['maskmem_pos_enc'],
            'obj_ids_list': self.all_obj_ids
            }
        #negative_masks_by_obj = self._normalize_negative_masks(negative_masks)

        # n_runs tells how many times we need to call the track function
        # this is useful especially in MOT setup, where few hundreds of objects
        # is tracked at the same time
        # in VOT the number of objects is much lower (up to 10)
        # which means that n_runs is always 1
        n_runs = ((len(output_dict_['obj_ids_list']) - 1) // self.max_batch_sz) + 1

        current_out = None  # output structure to collect (concatenate) outputs from multiple runs
        for i in range(n_runs):
            start_obj_idx = i * self.max_batch_sz
            end_obj_idx = min(len(output_dict_['obj_ids_list']),
                                i * self.max_batch_sz + self.max_batch_sz)

            obj_ids_list_ = output_dict_['obj_ids_list'][start_obj_idx:end_obj_idx]
            per_obj_dict_ = {}
            per_obj_obj_ptr_dict_ = {}
            for id_ in obj_ids_list_:
                per_obj_dict_[id_] = output_dict_['per_obj_dict'][id_]
                per_obj_obj_ptr_dict_[id_] = output_dict_['per_obj_obj_ptr_dict'][id_]
            output_dict_tmp = {'per_obj_dict': per_obj_dict_,
                               'per_obj_obj_ptr_dict': per_obj_obj_ptr_dict_,
                               'maskmem_pos_enc': output_dict_['maskmem_pos_enc'],
                               'obj_ids_list': obj_ids_list_}

            current_out_tmp = self.sam.track_step(
                frame_idx=self.frame_index,
                is_init_cond_frame=False,
                current_vision_feats=feats,
                current_vision_pos_embeds=pos,
                feat_sizes=feat_sizes,
                image=img,
                point_inputs=None,
                mask_inputs=None,
                output_dict=output_dict_tmp,
                num_frames=self.n_frames,
                track_in_reverse=False,
                run_mem_encoder=True,
                prev_sam_mask_logits=None,
            )

            current_out_tmp['maskmem_pos_enc'] = None

            # this if is here only to support multi-run setup (when huge number of objects is tracked)
            if current_out is None:
                current_out = current_out_tmp
            else:
                current_out['pred_masks'] = torch.cat([current_out['pred_masks'], current_out_tmp['pred_masks']], 0)
                current_out['obj_ptr'] = torch.cat([current_out['obj_ptr'], current_out_tmp['obj_ptr']], 0)
                current_out['object_score_logits'] = torch.cat([current_out['object_score_logits'], current_out_tmp['object_score_logits']], 0)
                current_out['maskmem_features'] = torch.cat([current_out['maskmem_features'], current_out_tmp['maskmem_features']], 0)

        return current_out

    def _postprocess_predictions(self, current_out):
        """Turn raw SAM3 outputs into full-resolution masks and derived arrays.

        Read-only w.r.t. tracker state (reads self.fill_hole_area,
        self.img_height, self.img_width). Returns:
            pred_masks_gpu       - low-res mask logits ([N_obj, 1, 256, 256])
            m                    - full-res binary masks (list of np.uint8)
            n_pixels_pos         - positive pixel counts per object
            maskmem_features     - bfloat16 memory features
            alternative_masks_all- full-res alternative masks (np.uint8)
            all_ious             - predicted IoUs (np.float)
        """
        pred_masks_gpu = current_out["pred_masks"]  # [N_obj, 1, 256, 256]
        # potentially fill holes in the predicted masks
        if self.fill_hole_area > 0:
            pred_masks_gpu = fill_holes_in_mask_scores(
                pred_masks_gpu, self.fill_hole_area
            )

        sz_ = (self.img_height, self.img_width)
        masks_out = torch.nn.functional.interpolate(pred_masks_gpu, size=sz_, mode="bilinear", align_corners=False)
        if torch.isnan(masks_out).any():
            print("FATAL ERROR: NaN detected in SAM 3 mask logits!", flush=True)
        m = [(m_[0] > 0).float().cpu().numpy().astype(np.uint8) for m_ in masks_out]
        n_pixels_pos = [m_single.sum() for m_single in m]

        maskmem_features = current_out["maskmem_features"].to(torch.bfloat16)

        alternative_masks_all = torch.nn.functional.interpolate(current_out["multimasks_logits"], size=sz_, mode="bilinear", align_corners=False)
        alternative_masks_all = (alternative_masks_all > 0).detach().cpu().numpy().astype(np.uint8)
        all_ious = current_out["ious"].detach().float().cpu().numpy()

        return (pred_masks_gpu, m, n_pixels_pos, maskmem_features,
                alternative_masks_all, all_ious)

    def track(self, image):
        self.frame_index += 1

        # prepare image
        img = self._prepare_image(image)
        img = img.unsqueeze(0)  # (1, 3, 1024, 1024)
        
        # compute features
        feats, pos, feat_sizes = self._get_features(img)  # Note: removed number of objects
        
        current_out = self._run_sam3_inference(img, feats, pos, feat_sizes)

        (pred_masks_gpu, m, n_pixels_pos, maskmem_features,
         alternative_masks_all, all_ious) = self._postprocess_predictions(current_out)

        memory_lock_selection = self._resolve_memory_lock_selection(
            alternative_masks_all, all_ious
        ) if len(self.all_obj_ids) > 1 else [
            {
                "candidate_idx": int(np.argmax(all_ious[obj_idx])),
                "mask": np.asarray(alternative_masks_all[obj_idx][int(np.argmax(all_ious[obj_idx]))]).astype(np.uint8),
                "pred_iou": float(np.max(all_ious[obj_idx])),
            }
            for obj_idx in range(len(self.all_obj_ids))
        ]
        selected_mask_idx_by_obj_idx = []
        selected_pred_iou_by_obj_idx = []
        selected_mask_is_resolved_by_obj_idx = []
        selected_mask_is_alternative_by_obj_idx = []
        selected_n_pixels_pos_by_obj_idx = []
        for obj_idx in range(len(self.all_obj_ids)):
            lock_choice = memory_lock_selection[obj_idx] if obj_idx < len(memory_lock_selection) else {
                "candidate_idx": None,
                "mask": None,
                "pred_iou": 0.0,
            }
            chosen_mask_idx = int(np.argmax(all_ious[obj_idx]))
            selected_mask_idx = lock_choice["candidate_idx"]
            selected_mask_is_resolved = selected_mask_idx is not None and lock_choice["mask"] is not None
            if selected_mask_idx is None:
                selected_mask_idx = chosen_mask_idx
            selected_mask_idx_by_obj_idx.append(int(selected_mask_idx))
            selected_mask_is_resolved_by_obj_idx.append(bool(selected_mask_is_resolved))
            selected_mask_is_alternative_by_obj_idx.append(
                bool(selected_mask_is_resolved and int(selected_mask_idx) != chosen_mask_idx)
            )
            selected_pred_iou_by_obj_idx.append(
                float(lock_choice["pred_iou"]) if selected_mask_is_resolved else 0.0
            )
            selected_n_pixels_pos_by_obj_idx.append(
                int(np.asarray(lock_choice["mask"]).sum())
                if selected_mask_is_resolved
                else int(n_pixels_pos[obj_idx])
            )

        overlap_info = {}
        overlap_info_valid = False
        low_obj_max_overlap = {obj_id: 0.0 for obj_id in self.all_obj_ids}

        # 计算每个mask的最大连通域
        if len(self.all_obj_ids) > 1:
            m_largest = [
                keep_largest_component(m_single) if m_single.sum() > 0 else m_single
                for m_single in m
            ]
            base_shape = m_largest[0].shape if m_largest else None
            shapes_aligned = all(mask.shape == base_shape for mask in m_largest)
            overlap_info_valid = shapes_aligned

            # 计算两两之间的重叠比值
            if shapes_aligned:
                for obj_idx_i in range(len(self.all_obj_ids)):
                    for obj_idx_j in range(obj_idx_i + 1, len(self.all_obj_ids)):
                        obj_id_i = self.all_obj_ids[obj_idx_i]
                        obj_id_j = self.all_obj_ids[obj_idx_j]
                        if not self._objects_overlap_visible(obj_id_i, obj_id_j):
                            continue

                        m_i_largest_count = m_largest[obj_idx_i].sum()
                        m_j_largest_count = m_largest[obj_idx_j].sum()

                        if m_i_largest_count == 0 or m_j_largest_count == 0:
                            continue

                        # 计算交集
                        intersection = np.logical_and(
                            m_largest[obj_idx_i], m_largest[obj_idx_j]
                        ).astype(np.uint8)
                        intersection_count = intersection.sum()

                        if intersection_count > 0:
                            # 计算与obj_i和obj_j的比值
                            ratio_i = intersection_count / m_i_largest_count
                            ratio_j = intersection_count / m_j_largest_count

                            overlap_info[(obj_id_i, obj_id_j)] = {
                                'intersection': intersection_count,
                                'ratio_i': ratio_i,
                                'ratio_j': ratio_j
                            }

        # Overlap-aware memory update freeze gate (hysteresis).
        # Only affects the lower object_score_logits object in each overlapping pair.
        if not hasattr(self, "overlap_update_freeze_state"):
            self.overlap_update_freeze_state = {}
        overlap_update_freeze_high_th = getattr(self, "overlap_update_freeze_high_th", 0.9)
        overlap_update_freeze_low_th = getattr(self, "overlap_update_freeze_low_th", 0.2)

        if len(self.all_obj_ids) > 1 and overlap_info_valid:
            obj_score_by_id = {
                obj_id: float(current_out["object_score_logits"][obj_idx].item())
                for obj_idx, obj_id in enumerate(self.all_obj_ids)
            }

            for (obj_id_i, obj_id_j), info in overlap_info.items():
                score_i = obj_score_by_id.get(obj_id_i, None)
                score_j = obj_score_by_id.get(obj_id_j, None)
                if score_i is None or score_j is None:
                    continue
                if score_i < score_j:
                    low_obj_id = obj_id_i
                    low_ratio = float(info.get("ratio_i", 0.0))
                elif score_j < score_i:
                    low_obj_id = obj_id_j
                    low_ratio = float(info.get("ratio_j", 0.0))
                else:
                    continue

                if low_ratio > low_obj_max_overlap.get(low_obj_id, 0.0):
                    low_obj_max_overlap[low_obj_id] = low_ratio

            for obj_id in self.all_obj_ids:
                frozen = bool(self.overlap_update_freeze_state.get(obj_id, False))
                max_overlap = float(low_obj_max_overlap.get(obj_id, 0.0))
                if (not frozen) and max_overlap >= overlap_update_freeze_high_th:
                    self.overlap_update_freeze_state[obj_id] = True
                elif frozen and max_overlap < overlap_update_freeze_low_th:
                    self.overlap_update_freeze_state[obj_id] = False

        alternative_masks_to_return = []
        vt_drm_sources = []
        vt_spawn_allowed_this_frame = not any(
            selected_mask_is_alternative_by_obj_idx[obj_idx]
            and self.all_obj_ids[obj_idx] in self.real_obj_ids
            for obj_idx in range(len(self.all_obj_ids))
        )
        for obj_idx, obj_id in enumerate(self.all_obj_ids):
            obj_mem = self.per_object_outputs_all[obj_id]
            obj_mem_obj_ptr = self.per_object_obj_ptr[obj_id]
            lock_choice = memory_lock_selection[obj_idx] if obj_idx < len(memory_lock_selection) else {
                "candidate_idx": None,
                "mask": None,
                "pred_iou": 0.0,
            }

            # for alternative masks debug
            selected_mask_idx = selected_mask_idx_by_obj_idx[obj_idx]
            alternative_masks = [
                mask for i, mask in enumerate(alternative_masks_all[obj_idx])
                if i != selected_mask_idx and float(all_ious[obj_idx][i]) > 0.7
            ]
            alternative_masks_to_return.append([np.asarray(mask).squeeze().astype(np.uint8) for mask in alternative_masks])


            # check if DRM has to be updated from the element from previous frame
            if self.add_to_drm_next[obj_id]:
                obj_mem = self.per_object_outputs_all[obj_id]
                obj_mem[-1] = self.add_to_drm_next[obj_id]
                self.add_to_drm_next[obj_id] = None
                drm_idxs = [mem_idx for mem_idx, mem_el in enumerate(obj_mem) if (not mem_el['is_init'] and mem_el['is_drm'])]
                if len(drm_idxs) > self.max_drm:
                    # remove from DRM if more than max DRM elements
                    obj_mem.pop(drm_idxs[0])

            cur_update_delta = self._get_current_update_delta(obj_mem)

            obj_score = current_out["object_score_logits"][obj_idx].item()
            max_pred_iou = float(selected_pred_iou_by_obj_idx[obj_idx])
            score_iou = obj_score * max_pred_iou
            ram_frame_candidates = [
                mem_el["frame_idx"] for mem_el in obj_mem
                if (not mem_el["is_init"] and not mem_el["is_drm"])
            ]
            last_ram_frame = max(ram_frame_candidates) if len(ram_frame_candidates) > 0 else None
            update_gate, update_gate_stats = self._get_adaptive_update_gate(
                obj_idx=obj_idx,
                n_pixels_pos=selected_n_pixels_pos_by_obj_idx[obj_idx],
                max_pred_iou=max_pred_iou,
                obj_score=obj_score,
                score_iou=score_iou,
                last_ram_frame=last_ram_frame,
            )
            self._record_score_iou_observation(obj_idx, score_iou, obj_score)

            has_resolved_mask = bool(selected_mask_is_resolved_by_obj_idx[obj_idx])
            was_frozen = bool(self.overlap_update_freeze_state.get(obj_id, False))
            if has_resolved_mask:
                self.overlap_update_freeze_state[obj_id] = False
            elif was_frozen:
                update_gate = False
            # update only if object is visible
            #if n_pixels_pos[obj_idx] > 0 and obj_score > 3.0:
            if update_gate:
                # list with all memory elements for this object
                obj_mem = self.per_object_outputs_all[obj_id]

                selected_mask_np = memory_lock_selection[obj_idx]["mask"]
                if selected_mask_np is None:
                    self.overlap_update_freeze_state[obj_id] = True
                    update_gate = False
                else:
                    selected_mask_lowres = mask_to_lowres_tensor(
                        selected_mask_np,
                        pred_masks_gpu.shape[-2:],
                        pred_masks_gpu.device,
                    )
                    if selected_mask_is_alternative_by_obj_idx[obj_idx]:
                        if self.fill_hole_area > 0:
                            selected_mask_lowres = fill_holes_in_mask_scores(
                                selected_mask_lowres, self.fill_hole_area
                            )
                        selected_mask_high_res = torch.nn.functional.interpolate(
                            selected_mask_lowres.to(img.device, non_blocking=True),
                            size=(self.sam.image_size, self.sam.image_size),
                            mode="bilinear",
                            align_corners=False,
                        )
                        selected_maskmem_features, _ = self.sam._encode_new_memory(
                            image=img,
                            current_vision_feats=feats,
                            feat_sizes=feat_sizes,
                            pred_masks_high_res=selected_mask_high_res,
                            object_score_logits=current_out["object_score_logits"][obj_idx].unsqueeze(0),
                            is_mask_from_pts=False,
                        )
                        selected_maskmem_features = selected_maskmem_features.to(torch.bfloat16)
                        selected_maskmem_features = selected_maskmem_features.to(img.device, non_blocking=True)

                        # Re-generate obj_ptr for alternative mask using track_step
                        alternative_mask_input = torch.nn.functional.interpolate(
                            selected_mask_lowres,
                            size=(self.sam.image_size, self.sam.image_size),
                            mode="bilinear",
                            align_corners=False,
                        )
                        alternative_mask_input = (alternative_mask_input >= 0.5).float()

                        alternative_out = self.sam.track_step(
                            frame_idx=self.frame_index,
                            is_init_cond_frame=False,
                            current_vision_feats=feats,
                            current_vision_pos_embeds=pos,
                            feat_sizes=feat_sizes,
                            image=img,
                            point_inputs=None,
                            mask_inputs=alternative_mask_input,
                            output_dict={
                                'per_obj_dict': {obj_id: self.per_object_outputs_all[obj_id]},
                                'per_obj_obj_ptr_dict': {obj_id: self.per_object_obj_ptr[obj_id]},
                                'maskmem_pos_enc': self.output_dict['maskmem_pos_enc'],
                                'obj_ids_list': [obj_id]
                            },
                            num_frames=self.n_frames,
                            track_in_reverse=False,
                            run_mem_encoder=False,
                            prev_sam_mask_logits=None,
                        )
                        selected_obj_ptr = alternative_out["obj_ptr"]
                    else:
                        selected_maskmem_features = maskmem_features[obj_idx].unsqueeze(0)
                        selected_obj_ptr = current_out["obj_ptr"][obj_idx].unsqueeze(0)
                    self._record_memory_update_trigger(obj_idx, score_iou, obj_score)

                    # Update object pointers with the correct pointer (alternative or original)
                    per_obj_obj_ptr_dict = {"obj_ptr": selected_obj_ptr,
                                            "frame_idx": self.frame_index, "is_init": False}
                    obj_mem_obj_ptr = self.per_object_obj_ptr[obj_id]
                    obj_mem_obj_ptr.append(per_obj_obj_ptr_dict)
                    if len(obj_mem_obj_ptr) > self.sam.max_obj_ptrs_in_encoder:
                        # get first non-init frame and remove it from the list
                        rem_idx = None
                        for i, ptr_el in enumerate(obj_mem_obj_ptr):
                            if not ptr_el["is_init"]:
                                rem_idx = i
                                break
                        if rem_idx:
                            obj_mem_obj_ptr.pop(rem_idx)

                    # Here the per-object update is performed
                    # create object dict and append it to list
                    per_obj_dict = {
                        "maskmem_features": selected_maskmem_features,  # (1, 64, 64, 64)
                        "pred_masks": selected_mask_lowres.detach().cpu().numpy(),  # (1, 1, 256, 256)
                        "is_init": False, "frame_idx": self.frame_index, "is_drm": False
                    }

                    if self.use_last:
                        ram_idxs = [mem_idx for mem_idx, mem_el in enumerate(obj_mem) if (not mem_el['is_init'] and not mem_el['is_drm'])]
                        
                        if len(ram_idxs) == 0:
                            obj_mem.append(per_obj_dict)
                        elif (self.frame_index % cur_update_delta) == 0:
                            if (obj_mem[ram_idxs[-1]]['frame_idx'] % cur_update_delta) == 0:
                                obj_mem.append(per_obj_dict)
                            else:
                                obj_mem[ram_idxs[-1]] = per_obj_dict
                        else:
                            if (obj_mem[ram_idxs[-1]]['frame_idx'] % cur_update_delta) == 0:
                                obj_mem.append(per_obj_dict)
                            else:
                                obj_mem[ram_idxs[-1]] = per_obj_dict
                    else:
                        if (self.frame_index % cur_update_delta) == 0:
                            obj_mem.append(per_obj_dict)

                    # check if memory is full for this object
                    # remove the oldest non-init RAM element
                    ram_idxs = [mem_idx for mem_idx, mem_el in enumerate(obj_mem) if (not mem_el['is_init'] and not mem_el['is_drm'])]
                    if len(ram_idxs) > self.max_ram and len(obj_mem) > self.sam.num_maskmem:
                        obj_mem.pop(ram_idxs[0])
                    
                    # update the DRM memory - but first, check if DRM is even in use
                    if self.max_drm > 0:
                        # check for update the DRM part of the memory
                        m_idx = int(selected_mask_idx_by_obj_idx[obj_idx])
                        m_iou = float(lock_choice["pred_iou"])
                        # Delete the chosen predicted mask from the list of all predicted masks, leading to only alternative masks
                        alternative_masks = [mask for i, mask in enumerate(alternative_masks_all[obj_idx]) if i != m_idx]

                        # Determine if the object ratio between the current frame and the previous frames is within a certain range
                        self.object_sizes[obj_idx].append(
                            selected_n_pixels_pos_by_obj_idx[obj_idx]
                        )
                        if len(self.object_sizes[obj_idx]) > 1:
                            obj_sizes_ratio = selected_n_pixels_pos_by_obj_idx[obj_idx] / np.median([
                                size for size in self.object_sizes[obj_idx][-300:] if size >= 1
                            ][-10:])
                        else:
                            obj_sizes_ratio = -1

                        # The first condition checks if:
                        #  - the chosen predicted mask has a high predicted IoU, 
                        #  - the object size ratio is within a +- 20% range compared to the previous frames, 
                        #  - the target is present in the current frame,
                        #  - the last added frame to DRM is more than 5 frames ago or no frame has been added yet
                        if m_iou > 0.8 and obj_sizes_ratio >= 0.8 and obj_sizes_ratio <= 1.2 and \
                            (self.frame_index - self.last_added[obj_idx] > cur_update_delta or self.last_added[obj_idx] == -1):
                            # Numpy array of the chosen mask and corresponding bounding box
                            chosen_mask_np = selected_mask_np
                            chosen_bbox = npmask2box(chosen_mask_np)

                            # Delete the parts of the alternative masks that overlap with the chosen mask
                            alternative_masks = [np.logical_and(m_, np.logical_not(chosen_mask_np)).astype(np.uint8) for m_ in alternative_masks]
                            # Keep only the largest connected component of the processed alternative masks
                            alternative_masks = [keep_largest_component(m_) for m_ in alternative_masks if np.sum(m_) >= 1]
                            if len(alternative_masks) > 0:
                                # Make the union of the chosen mask and the processed alternative masks (corresponding to the largest connected component)
                                alternative_masks = [np.logical_or(m_, chosen_mask_np).astype(np.uint8) for m_ in alternative_masks]
                                # Convert the processed alternative masks to bounding boxes to calculate the IoUs bounding box-wise
                                alternative_bboxes = [npmask2box(m_) for m_ in alternative_masks]
                                # Calculate the IoUs between the chosen bounding box and the processed alternative bounding boxes
                                ious = [calculate_overlaps([Rectangle(*chosen_bbox)], [Rectangle(*bbox)])[0] for bbox in alternative_bboxes]
                                # The second condition checks if within the calculated IoUs, there is at least one IoU that is less than 0.7
                                # That would mean that there are significant differences between the chosen mask and the processed alternative masks, 
                                # leading to possible detections of distractors within alternative masks.
                                if np.min(np.array(ious)) <= 0.7:
                                    self.last_added[obj_idx] = self.frame_index # Update the last added frame index
                                    alternative_evidence = []
                                    if vt_spawn_allowed_this_frame and not selected_mask_is_alternative_by_obj_idx[obj_idx]:
                                        alternative_evidence = self._build_vt_alternative_evidence(
                                            [mask for i, mask in enumerate(alternative_masks_all[obj_idx]) if i != m_idx],
                                            chosen_mask_np,
                                        )
                                    if obj_id in self.real_obj_ids and alternative_evidence:
                                        x, y, w, h = chosen_bbox
                                        vt_drm_sources.append({
                                            "obj_id": obj_id,
                                            "box": [
                                                (x + 0.5 * w) / self.img_width,
                                                (y + 0.5 * h) / self.img_height,
                                                w / self.img_width,
                                                h / self.img_height,
                                            ],
                                            "center": (x + 0.5 * w, y + 0.5 * h),
                                            "alternative_evidence": alternative_evidence,
                                        })
                                
                                    # add element to DRM
                                    per_obj_dict = {
                                        "maskmem_features": selected_maskmem_features,  # (1, 64, 64, 64)
                                        "pred_masks": selected_mask_lowres.detach().cpu().numpy(),  # (1, 1, 256, 256)
                                        "is_init": False, "frame_idx": self.frame_index, "is_drm": True
                                    }
                                
                                    if self.frame_index == obj_mem[-1]['frame_idx']:
                                        # this frame is already in RAM; 
                                        # put into the temporary mem structure and add to DRM in the next frame
                                        self.add_to_drm_next[obj_id] = per_obj_dict
                                    else:
                                        # this frame is not in RAM yet - add directly to DRM
                                        obj_mem.append(per_obj_dict)
                                    
                                        # check if memory is full for this object
                                        # remove the oldest non-init DRM element
                                        if len(obj_mem) > self.sam.num_maskmem:
                                            drm_idxs = [mem_idx for mem_idx, mem_el in enumerate(obj_mem) if (not mem_el['is_init'] and mem_el['is_drm'])]
                                            if len(drm_idxs) > self.max_drm:
                                                # remove from DRM if more than max DRM elements
                                                obj_mem.pop(drm_idxs[0])
                                            else:
                                                # remove from RAM elsewhere
                                                ram_idxs = [mem_idx for mem_idx, mem_el in enumerate(obj_mem) if (not mem_el['is_init'] and not mem_el['is_drm'])]
                                                obj_mem.pop(ram_idxs[0])
            
        tracked_obj_ids = list(self.all_obj_ids)
            
        if vt_spawn_allowed_this_frame:
            self._maybe_spawn_virtual_trackers(
                image=image,
                masks=m,
                drm_sources=vt_drm_sources,
                img=img,
                feats=feats,
                pos=pos,
                feat_sizes=feat_sizes,
            )
        virtual_obj_ids_to_remove = []
        for obj_idx, obj_id in enumerate(tracked_obj_ids):
            if obj_id not in self.virtual_obj_ids:
                continue
            if n_pixels_pos[obj_idx] == 0:
                self.virtual_empty_frame_count[obj_id] = (
                    self.virtual_empty_frame_count.get(obj_id, 0) + 1
                )
            else:
                self.virtual_empty_frame_count[obj_id] = 0
            if self.virtual_empty_frame_count[obj_id] > self.vt_empty_mask_patience:
                virtual_obj_ids_to_remove.append(obj_id)
        if virtual_obj_ids_to_remove:
            self._remove_virtual_trackers(virtual_obj_ids_to_remove)

        output_masks_by_obj_idx = [
            np.asarray(mask).squeeze().astype(np.uint8).copy()
            for mask in m
        ]
        for obj_idx in range(len(tracked_obj_ids)):
            if selected_mask_is_alternative_by_obj_idx[obj_idx]:
                selected_mask_np = memory_lock_selection[obj_idx]["mask"]
                if selected_mask_np is not None:
                    output_masks_by_obj_idx[obj_idx] = (
                        np.asarray(selected_mask_np).squeeze().astype(np.uint8).copy()
                    )

        # If an object's memory update is frozen due to overlap, suppress its output mask.
        # This is applied right before returning outputs to avoid affecting any earlier logic.
        if getattr(self, "overlap_update_freeze_state", None):
            for obj_idx, obj_id in enumerate(tracked_obj_ids):
                if self.overlap_update_freeze_state.get(obj_id, False):
                    try:
                        output_masks_by_obj_idx[obj_idx][...] = 0
                    except Exception:
                        output_masks_by_obj_idx[obj_idx] = np.zeros_like(
                            output_masks_by_obj_idx[obj_idx]
                        )
        real_outputs = []
        real_alternative_outputs = []
        for obj_id in self.real_obj_ids:
            if obj_id not in tracked_obj_ids:
                continue
            obj_idx = tracked_obj_ids.index(obj_id)
            real_outputs.append(output_masks_by_obj_idx[obj_idx])
            real_alternative_outputs.append(alternative_masks_to_return[obj_idx])

        outputs = {'masks': real_outputs, 'alternative_masks': real_alternative_outputs}
        return outputs
