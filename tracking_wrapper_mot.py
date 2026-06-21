import os
import sys
import types
import numpy as np
import cv2
import torch
import torch.nn as nn
import time

from collections import OrderedDict

import logging
from hydra import compose
from hydra.utils import instantiate
from omegaconf import OmegaConf

from vot.region.raster import calculate_overlaps
from vot.region.shapes import Rectangle


from sam3.model.sam3_tracker_utils import fill_holes_in_mask_scores
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
    return tracker

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

def load_confs(chkpt_path, model_size):
    checkpoint = os.path.join(chkpt_path, 'sam3.pt')
    #checkpoint = os.path.join(chkpt_path, 'sam3.1_multiplex.pt')
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"SAM 3 checkpoint not found: {checkpoint}")
        #raise FileNotFoundError(f"SAM 3.1 checkpoint not found: {checkpoint}")
    return checkpoint


class DAM4SAMMOT():
    def __init__(self, model_size='large', checkpoint_dir=None, offload_state_to_cpu=False):
        
        if not checkpoint_dir:
            checkpoint_dir = './checkpoints'
        checkpoint = load_confs(checkpoint_dir, model_size)
        self.device = _env_device("D4SM3_DEVICE", _default_cuda_device(prefer_second=True))
        self.sam = build_sam(checkpoint, device=str(self.device))

        self.input_image_size = 1008
        self.fill_hole_area = 8
        
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

        self.device = self.device
        if offload_state_to_cpu:
            self.storage_device = torch.device("cpu")
        else:
            self.storage_device = self.device
        
        self.non_overlap_masks_for_mem_enc = False
        self.binarize_mask_from_pts_for_mem_enc = True

        # MOT-specific fields
        self.per_object_outputs_all = {}
        self.per_object_obj_ptr = {}  # separate object pointers since they are updated differently
        self.next_obj_id = 1
        self.all_obj_ids = []
        self.max_batch_sz = 200  # how many objects will be processed together (should not impact tracking)
        self.update_delta = 5  # update every delta frames
        self.max_ram = 3
        self.max_drm = 3
        self.use_last = True  # always use last frame in RAM
        self.add_to_drm_next = {}  # needed for DRM update (to prevent adding twice the same frame to the memory)

        self.use_adaptive_update_gate = True
        self.update_gate_window = 20
        self.update_gate_moving_window = 10
        self.update_gate_warmup = 3
        self.update_gate_ema_alpha = 0.1
        self.update_gate_score_iou_floor = 0.0
        self.update_gate_score_iou_cap = 6.0
        self.update_gate_ref_ratio = 0.35
        self.update_gate_last_ratio = 0.5
        # "accepted" computes moving_avg/EMA only from frames that pass update_gate.
        # Set to "all" to restore the previous behavior: every positive object score.
        self.update_gate_stats_source = os.environ.get(
            "SAFE_DAM_UPDATE_GATE_STATS_SOURCE", "all"
        ).lower()
        self.enable_high_iou_rescue = False
        self.update_gate_high_iou_rescue = 0.9
        self.update_gate_rescue_ratio = 0.5
        self.update_gate_min_pred_iou = 0.0
        self.update_gate_min_obj_score = 0.0
        self.use_update_gate_continuation_rescue = False
        self.update_gate_continuation_max_gap = 1
        self.update_gate_continuation_score_iou_floor = 1.0
        self.update_gate_continuation_min_pred_iou = 0.65
        self.update_gate_continuation_min_obj_score = 1.4
        self.update_gate_max_history = 300
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
        self.add_to_drm_next = {}
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
    
    def _npmask2box(self, mask):
        # mask is a 2D numpy array in a np.uint8 format
        x_ = np.where(mask.sum(0) > 0)[0]
        y_ = np.where(mask.sum(1) > 0)[0]
        x0, x1 = x_.min(), x_.max()
        y0, y1 = y_.min(), y_.max()
        # convert to (x, y0, width, height) bbox format
        return [x0, y0, x1 - x0 + 1, y1 - y0 + 1]
    """
    def _normalize_negative_masks(self, negative_masks):
        if negative_masks is None:
            return {}
        if isinstance(negative_masks, dict):
            return negative_masks
        if isinstance(negative_masks, (list, tuple)):
            if len(negative_masks) != len(self.all_obj_ids):
                raise ValueError(
                    "negative_masks as a list/tuple must be aligned with all_obj_ids "
                    f"({len(self.all_obj_ids)} objects, got {len(negative_masks)} masks)"
                )
            return {
                obj_id: mask
                for obj_id, mask in zip(self.all_obj_ids, negative_masks)
                if mask is not None
            }
        return {obj_id: negative_masks for obj_id in self.all_obj_ids}

    def _mask_to_numpy_bool(self, mask):
        if isinstance(mask, torch.Tensor):
            mask = mask.detach().cpu().numpy()
        mask = np.asarray(mask)
        if mask.ndim == 3:
            mask = np.squeeze(mask)
        if mask.ndim != 2:
            raise ValueError(f"Expected a 2D negative mask, got shape {mask.shape}")
        if mask.shape != (self.img_height, self.img_width):
            mask = cv2.resize(
                mask.astype(np.float32),
                (self.img_width, self.img_height),
                interpolation=cv2.INTER_NEAREST,
            )
        return mask.astype(bool)

    def _sample_points_from_binary_mask(self, mask, num_points):
        if num_points <= 0 or not np.any(mask):
            return []

        mask_u8 = mask.astype(np.uint8)
        points = []
        work_mask = mask_u8.copy()
        for _ in range(num_points):
            if not np.any(work_mask):
                break
            dist = cv2.distanceTransform(work_mask, cv2.DIST_L2, 5)
            _, max_val, _, max_loc = cv2.minMaxLoc(dist)
            if max_val <= 0:
                ys, xs = np.where(work_mask > 0)
                mid = len(xs) // 2
                points.append((float(xs[mid]), float(ys[mid])))
                break
            x, y = max_loc
            points.append((float(x), float(y)))
            radius = max(3, int(max_val * 0.75))
            cv2.circle(work_mask, (x, y), radius, 0, thickness=-1)
        return points

    def _build_negative_point_inputs(
        self,
        obj_ids_list,
        pred_masks_low_res,
        negative_masks_by_obj,
        negative_points_per_mask=4,
        positive_points_per_mask=1,
    ):
        device = pred_masks_low_res.device
        batch_size = len(obj_ids_list)
        max_points = max(1, negative_points_per_mask + positive_points_per_mask)
        coords = torch.zeros(batch_size, max_points, 2, dtype=torch.float32, device=device)
        labels = -torch.ones(batch_size, max_points, dtype=torch.int32, device=device)

        pred_masks_orig = torch.nn.functional.interpolate(
            pred_masks_low_res.detach(),
            size=(self.img_height, self.img_width),
            mode="bilinear",
            align_corners=False,
        )
        pred_masks_orig = (pred_masks_orig[:, 0] > 0).detach().cpu().numpy()

        any_prompt = False
        for batch_idx, obj_id in enumerate(obj_ids_list):
            neg_mask = negative_masks_by_obj.get(obj_id, None)
            if neg_mask is None:
                continue
            neg_mask = self._mask_to_numpy_bool(neg_mask)
            if not np.any(neg_mask):
                continue

            point_list = []
            label_list = []
            safe_pos_mask = np.logical_and(pred_masks_orig[batch_idx], np.logical_not(neg_mask))
            for x, y in self._sample_points_from_binary_mask(safe_pos_mask, positive_points_per_mask):
                point_list.append((x, y))
                label_list.append(1)
            for x, y in self._sample_points_from_binary_mask(neg_mask, negative_points_per_mask):
                point_list.append((x, y))
                label_list.append(0)

            if len(point_list) == 0:
                continue

            any_prompt = True
            for point_idx, ((x, y), label) in enumerate(zip(point_list[:max_points], label_list[:max_points])):
                coords[batch_idx, point_idx, 0] = x / self.img_width * self.sam.image_size
                coords[batch_idx, point_idx, 1] = y / self.img_height * self.sam.image_size
                labels[batch_idx, point_idx] = label

        if not any_prompt:
            return None
        return {"point_coords": coords, "point_labels": labels}

    def _suppress_negative_masks_in_output(
        self,
        current_out,
        obj_ids_list,
        negative_masks_by_obj,
        img,
        feats,
        feat_sizes,
        output_dict,
        is_init_cond_frame,
    ):
        if negative_masks_by_obj is None or len(negative_masks_by_obj) == 0:
            return current_out

        pred_masks = current_out["pred_masks"]
        low_res_suppression = torch.zeros_like(pred_masks, dtype=torch.bool)
        high_res_suppression = torch.zeros_like(current_out["multimasks_logits"], dtype=torch.bool)

        for batch_idx, obj_id in enumerate(obj_ids_list):
            neg_mask = negative_masks_by_obj.get(obj_id, None)
            if neg_mask is None:
                continue
            neg_mask = self._mask_to_numpy_bool(neg_mask)
            neg_tensor = torch.from_numpy(neg_mask).to(pred_masks.device)[None, None].float()
            low_res_mask = torch.nn.functional.interpolate(
                neg_tensor,
                size=pred_masks.shape[-2:],
                mode="nearest",
            )[0, 0].bool()
            high_res_mask = torch.nn.functional.interpolate(
                neg_tensor,
                size=current_out["multimasks_logits"].shape[-2:],
                mode="nearest",
            )[0, 0].bool()
            low_res_suppression[batch_idx, 0] = low_res_mask
            high_res_suppression[batch_idx] = high_res_mask.expand_as(high_res_suppression[batch_idx])

        if not bool(low_res_suppression.any().item()):
            return current_out

        current_out["pred_masks"] = pred_masks.masked_fill(low_res_suppression, -32.0)
        current_out["multimasks_logits"] = current_out["multimasks_logits"].masked_fill(
            high_res_suppression, -32.0
        )
        current_out["pred_masks_high_res"] = current_out["pred_masks_high_res"].masked_fill(
            high_res_suppression[:, :1], -32.0
        )

        maskmem_features, maskmem_pos_enc = self.sam._encode_new_memory(
            image=img,
            current_vision_feats=feats,
            feat_sizes=feat_sizes,
            pred_masks_high_res=current_out["pred_masks_high_res"],
            object_score_logits=current_out["object_score_logits"],
            is_mask_from_pts=True,
            output_dict=output_dict,
            is_init_cond_frame=is_init_cond_frame,
        )
        current_out["maskmem_features"] = maskmem_features
        current_out["maskmem_pos_enc"] = maskmem_pos_enc
        return current_out
    """

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
            self.next_obj_id += 1
        
        return None
    """
    def track(
        self,
        image,
        negative_masks=None,
        negative_points_per_mask=4,
        positive_points_per_mask=1,
        suppress_negative_masks=True,
    ):
    """
    def track(self, image):
        self.frame_index += 1

        # prepare image
        img = self._prepare_image(image)
        img = img.unsqueeze(0)  # (1, 3, 1024, 1024)
        
        # compute features
        feats, pos, feat_sizes = self._get_features(img)  # Note: removed number of objects
        
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
            """
            point_inputs_ = self._build_negative_point_inputs(
                obj_ids_list=obj_ids_list_,
                pred_masks_low_res=current_out_tmp["pred_masks"],
                negative_masks_by_obj=negative_masks_by_obj,
                negative_points_per_mask=negative_points_per_mask,
                positive_points_per_mask=positive_points_per_mask,
            )
            if point_inputs_ is not None:
                prev_sam_mask_logits = torch.clamp(
                    current_out_tmp["pred_masks"].detach(), -32.0, 32.0
                )
                current_out_tmp = self.sam.track_step(
                    frame_idx=self.frame_index,
                    is_init_cond_frame=False,
                    current_vision_feats=feats,
                    current_vision_pos_embeds=pos,
                    feat_sizes=feat_sizes,
                    image=img,
                    point_inputs=point_inputs_,
                    mask_inputs=None,
                    output_dict=output_dict_tmp,
                    num_frames=self.n_frames,
                    track_in_reverse=False,
                    run_mem_encoder=True,
                    prev_sam_mask_logits=prev_sam_mask_logits,
                )
                if suppress_negative_masks:
                    current_out_tmp = self._suppress_negative_masks_in_output(
                        current_out=current_out_tmp,
                        obj_ids_list=obj_ids_list_,
                        negative_masks_by_obj=negative_masks_by_obj,
                        img=img,
                        feats=feats,
                        feat_sizes=feat_sizes,
                        output_dict=output_dict_tmp,
                        is_init_cond_frame=False,
                    )
            """
            current_out_tmp['maskmem_pos_enc'] = None

            # this if is here only to support multi-run setup (when huge number of objects is tracked)
            if current_out is None:
                current_out = current_out_tmp
            else:
                current_out['pred_masks'] = torch.cat([current_out['pred_masks'], current_out_tmp['pred_masks']], 0)
                current_out['obj_ptr'] = torch.cat([current_out['obj_ptr'], current_out_tmp['obj_ptr']], 0)
                current_out['object_score_logits'] = torch.cat([current_out['object_score_logits'], current_out_tmp['object_score_logits']], 0)
                current_out['maskmem_features'] = torch.cat([current_out['maskmem_features'], current_out_tmp['maskmem_features']], 0)
            
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

        overlap_info = {}
        overlap_info_valid = False
        
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
            else:
                print(
                    f"Frame {self.frame_index} Mask Overlap Warning: "
                    f"shape mismatch={ [mask.shape for mask in m_largest] }"
                )

            # 输出重叠信息
            if overlap_info:
                print(
                    f"Frame {self.frame_index} Mask Overlap Summary "
                    f"(largest connected components):"
                )
                
                for (obj_id_i, obj_id_j), info in overlap_info.items():
                    print(
                        f"  Obj {obj_id_i} <-> Obj {obj_id_j}: "
                        f"intersection={info['intersection']}, "
                        f"ratio_with_obj{obj_id_i}={info['ratio_i']:.4f}, "
                        f"ratio_with_obj{obj_id_j}={info['ratio_j']:.4f}"
                    )
            
        
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

            low_obj_max_overlap = {obj_id: 0.0 for obj_id in self.all_obj_ids}
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

        alternative_masks_all = torch.nn.functional.interpolate(current_out["multimasks_logits"], size=sz_, mode="bilinear", align_corners=False)
        alternative_masks_all = (alternative_masks_all > 0).detach().cpu().numpy().astype(np.uint8)
        all_ious = current_out["ious"].detach().float().cpu().numpy()
      
        alternative_masks_to_return = []
        for obj_idx, obj_id in enumerate(self.all_obj_ids):
            obj_mem = self.per_object_outputs_all[obj_id]
            obj_mem_obj_ptr = self.per_object_obj_ptr[obj_id]

            # for alternative masks debug
            chosen_mask_idx = int(np.argmax(all_ious[obj_idx]))
            alternative_masks = [mask for i, mask in enumerate(alternative_masks_all[obj_idx]) if i != chosen_mask_idx]
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
            obj_score = current_out["object_score_logits"][obj_idx].item()
            max_pred_iou = float(np.max(all_ious[obj_idx]))
            score_iou = obj_score * max_pred_iou
            ram_frame_candidates = [
                mem_el["frame_idx"] for mem_el in obj_mem
                if (not mem_el["is_init"] and not mem_el["is_drm"])
            ]
            last_ram_frame = max(ram_frame_candidates) if len(ram_frame_candidates) > 0 else None
            update_gate, update_gate_stats = self._get_adaptive_update_gate(
                obj_idx=obj_idx,
                n_pixels_pos=n_pixels_pos[obj_idx],
                max_pred_iou=max_pred_iou,
                obj_score=obj_score,
                score_iou=score_iou,
                last_ram_frame=last_ram_frame,
            )
            self._record_score_iou_observation(obj_idx, score_iou, obj_score)

            if self.overlap_update_freeze_state.get(obj_id, False):
                print(f"Frame {self.frame_index} Obj {obj_id} Update Frozen due to Overlap, ratio={low_obj_max_overlap.get(obj_id, 0.0):.4f}")
                update_gate = False
            # update only if object is visible
            #if n_pixels_pos[obj_idx] > 0 and obj_score > 3.0:
            if update_gate:
                self._record_memory_update_trigger(obj_idx, score_iou, obj_score)
                # list with all memory elements for this object
                obj_mem = self.per_object_outputs_all[obj_id]

                # Update object pointers firs
                per_obj_obj_ptr_dict = {"obj_ptr": current_out["obj_ptr"][obj_idx].unsqueeze(0), 
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
                    "maskmem_features": maskmem_features[obj_idx].unsqueeze(0),  # (1, 64, 64, 64)
                    "pred_masks": pred_masks_gpu[obj_idx].unsqueeze(0).detach().cpu().numpy(),  # (1, 1, 256, 256)
                    "is_init": False, "frame_idx": self.frame_index, "is_drm": False
                }

                if self.use_last:
                    ram_idxs = [mem_idx for mem_idx, mem_el in enumerate(obj_mem) if (not mem_el['is_init'] and not mem_el['is_drm'])]
                    
                    if len(ram_idxs) == 0:
                        obj_mem.append(per_obj_dict)
                    elif (self.frame_index % self.update_delta) == 0:
                        if (obj_mem[ram_idxs[-1]]['frame_idx'] % self.update_delta) == 0:
                            obj_mem.append(per_obj_dict)
                        else:
                            obj_mem[ram_idxs[-1]] = per_obj_dict
                    else:
                        if (obj_mem[ram_idxs[-1]]['frame_idx'] % self.update_delta) == 0:
                            obj_mem.append(per_obj_dict)
                        else:
                            obj_mem[ram_idxs[-1]] = per_obj_dict
                else:
                    if (self.frame_index % self.update_delta) == 0:
                        obj_mem.append(per_obj_dict)
                
                # check if memory is full for this object
                # remove the oldest non-init RAM element
                ram_idxs = [mem_idx for mem_idx, mem_el in enumerate(obj_mem) if (not mem_el['is_init'] and not mem_el['is_drm'])]
                if len(ram_idxs) > self.max_ram and len(obj_mem) > self.sam.num_maskmem:
                    obj_mem.pop(ram_idxs[0])
                
                # update the DRM memory - but first, check if DRM is even in use
                if self.max_drm > 0:
                    # check for update the DRM part of the memory
                    m_idx = np.argmax(all_ious[obj_idx]) # Index of the chosen predicted mask
                    m_iou = all_ious[obj_idx][m_idx] # Predicted IoU of the chosen predicted mask
                    # Delete the chosen predicted mask from the list of all predicted masks, leading to only alternative masks
                    alternative_masks = [mask for i, mask in enumerate(alternative_masks_all[obj_idx]) if i != m_idx]

                    # Determine if the object ratio between the current frame and the previous frames is within a certain range
                    self.object_sizes[obj_idx].append(n_pixels_pos[obj_idx])
                    if len(self.object_sizes[obj_idx]) > 1:
                        obj_sizes_ratio = n_pixels_pos[obj_idx] / np.median([
                            size for size in self.object_sizes[obj_idx][-300:] if size >= 1
                        ][-10:])
                    else:
                        obj_sizes_ratio = -1

                    # The first condition checks if:
                    #  - the chosen predicted mask has a high predicted IoU, 
                    #  - the object size ratio is within a +- 20% range compared to the previous frames, 
                    #  - the target is present in the current frame,
                    #  - the last added frame to DRM is more than 5 frames ago or no frame has been added yet
                    print(f"DRM CHECK: Frame {self.frame_index}, Obj {obj_id} DRM parameters: m_iou={m_iou:.4f}, obj_sizes_ratio={obj_sizes_ratio:.4f}, frame_index - last_added={self.frame_index - self.last_added[obj_idx]}, last_added={self.last_added[obj_idx]}, DRM first gate={m_iou > 0.8 and obj_sizes_ratio >= 0.8 and obj_sizes_ratio <= 1.2 and (self.frame_index - self.last_added[obj_idx] > self.update_delta or self.last_added[obj_idx] == -1)}")
                    if m_iou > 0.8 and obj_sizes_ratio >= 0.8 and obj_sizes_ratio <= 1.2 and \
                        (self.frame_index - self.last_added[obj_idx] > self.update_delta or self.last_added[obj_idx] == -1):
                        
                        # Numpy array of the chosen mask and corresponding bounding box
                        chosen_mask_np = m[obj_idx]
                        chosen_bbox = self._npmask2box(m[obj_idx])

                        # Delete the parts of the alternative masks that overlap with the chosen mask
                        alternative_masks = [np.logical_and(m_, np.logical_not(chosen_mask_np)).astype(np.uint8) for m_ in alternative_masks]
                        # Keep only the largest connected component of the processed alternative masks
                        alternative_masks = [keep_largest_component(m_) for m_ in alternative_masks if np.sum(m_) >= 1]
                        if len(alternative_masks) > 0:
                            # Make the union of the chosen mask and the processed alternative masks (corresponding to the largest connected component)
                            alternative_masks = [np.logical_or(m_, chosen_mask_np).astype(np.uint8) for m_ in alternative_masks]
                            # Convert the processed alternative masks to bounding boxes to calculate the IoUs bounding box-wise
                            alternative_bboxes = [self._npmask2box(m_) for m_ in alternative_masks]
                            # Calculate the IoUs between the chosen bounding box and the processed alternative bounding boxes
                            ious = [calculate_overlaps([Rectangle(*chosen_bbox)], [Rectangle(*bbox)])[0] for bbox in alternative_bboxes]
                            # The second condition checks if within the calculated IoUs, there is at least one IoU that is less than 0.7
                            # That would mean that there are significant differences between the chosen mask and the processed alternative masks, 
                            # leading to possible detections of distractors within alternative masks.
                            print(f"DRM CHECK: Frame {self.frame_index}, Obj {obj_id} DRM parameters: bbox divergence={np.min(np.array(ious)):.4f}, DRM trigger={np.min(np.array(ious)) <= 0.7}")
                            if np.min(np.array(ious)) <= 0.7:
                                self.last_added[obj_idx] = self.frame_index # Update the last added frame index
                                
                                # add element to DRM
                                per_obj_dict = {
                                    "maskmem_features": maskmem_features[obj_idx].unsqueeze(0),  # (1, 64, 64, 64)
                                    "pred_masks": pred_masks_gpu[obj_idx].unsqueeze(0).detach().cpu().numpy(),  # (1, 1, 256, 256)
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
            print(
                f"Memory update gate: Frame {self.frame_index}, Obj {obj_id}, "
                f"score_iou={score_iou:.4f}, threshold={update_gate_stats['threshold']:.4f}, "
                f"rescue_threshold={update_gate_stats['rescue_threshold']:.4f}, "
                f"accepted_median={self._format_gate_value(update_gate_stats['accepted_median'])}, "
                f"moving_avg={self._format_gate_value(update_gate_stats['moving_avg'])}, "
                f"stats_source={getattr(self, 'update_gate_stats_source', 'accepted')}, "
                f"last_score={self._format_gate_value(update_gate_stats['last_score'])}, "
                f"warmup={update_gate_stats['is_warmup']}, "
                f"score_gate={update_gate_stats['score_gate']}, "
                f"high_iou_rescue_gate={update_gate_stats['high_iou_rescue_gate']}, "
                f"continuation_rescue_gate={update_gate_stats['continuation_rescue_gate']}, "
                f"ram_gap={self._format_gate_value(update_gate_stats['ram_gap'])}, "
                f"update={update_gate}"
            )
            print(f"Frame {self.frame_index} Obj {obj_id} IoUs: {all_ious[obj_idx]}")
            print(f"Logits range: {current_out['multimasks_logits'][obj_idx].min()} to {current_out['multimasks_logits'][obj_idx].max()}")                        
            print(f"Object score logits: {current_out['object_score_logits'][obj_idx].item():.4f}")
            print(f"Current RAM List for Obj {obj_id}: {[mem_el['frame_idx'] for mem_el in obj_mem if (not mem_el['is_init'] and not mem_el['is_drm'])]}")
            print(f"Current DRM List for Obj {obj_id}: {[mem_el['frame_idx'] for mem_el in obj_mem if (not mem_el['is_init'] and mem_el['is_drm'])]}\n")

        # If an object's memory update is frozen due to overlap, suppress its output mask.
        # This is applied right before returning outputs to avoid affecting any earlier logic.
        if getattr(self, "overlap_update_freeze_state", None):
            for obj_idx, obj_id in enumerate(self.all_obj_ids):
                if self.overlap_update_freeze_state.get(obj_id, False):
                    try:
                        m[obj_idx][...] = 0
                    except Exception:
                        m[obj_idx] = np.zeros_like(m[obj_idx])
        outputs = {'masks': m, 'alternative_masks': alternative_masks_to_return}
        return outputs
