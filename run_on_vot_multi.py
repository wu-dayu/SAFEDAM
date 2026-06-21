import sys, os

current_dir = os.path.dirname(os.path.abspath(__file__))

sys.path.remove(current_dir)


sys.path.insert(0, current_dir)

import argparse

import time
import torch

import cv2
import torch
from PIL import Image
import numpy as np

import hydra

from vot.region import Rectangle, Mask, is_special
from vot.dataset import load_dataset

from tracking_wrapper_mot_VT_7701 import DAM4SAMMOT
from visualization_utils import Visualizer

import contextlib

"""
def vot2bbox(vot_region):
    if vot_region.type != RegionType.SPECIAL:
        bb = vot_region.convert(RegionType.RECTANGLE)
        if not bb.is_empty():
            return [bb.x, bb.y, bb.width, bb.height]
        return None
"""
def vot2bbox(vot_region):
    # 直接使用官方提供的 is_special() 判断是否为丢失/无效帧
    if not is_special(vot_region):
        try:
            # 新版 API 的标准转换方式：调用目标类的静态 convert 方法
            bb = Rectangle.convert(vot_region)
        except AttributeError:
            # 兼容性 Fallback：如果某些过渡版本不支持 convert，则提取 bounds
            if hasattr(vot_region, 'bounds'):
                bb = Rectangle(*vot_region.bounds())
            else:
                bb = vot_region
        
        # 返回边界框列表
        if not bb.is_empty():
            return [bb.x, bb.y, bb.width, bb.height]
            
    return None

@torch.inference_mode()
@torch.amp.autocast("cuda")
def run_sequence(dataset_path, sequence_name, checkpoint_path, visualize, save_vis_dir=None):
    dataset = load_dataset(dataset_path)

    seq_names = dataset.list()
    if sequence_name is not None:
        seq_names = [sequence_name]

    tracker = DAM4SAMMOT(model_size='large', checkpoint_dir=checkpoint_path)
    VALID_NAMES = {
        "babychimp", "balls", "birdflock", "bus-3", "chevrotain",
        "chimp", "deerchase", "doe-1", "doe-2", "dolphins-2",
        "ducklings", "ducks", "elephant", "goose", "keyboard",
        "lettercube", "macrons", "numbers", "piggy", "pinkchick",
        "safaribird", "sailboat", "sealion", "stock", "whitechick",
        "zebra"
    }
    for seq_name in seq_names:
        if seq_name in VALID_NAMES:
            continue
        #log_path = os.path.join("/bd_byt4090i0/users/omnimotion/dayu/vots_difficult/log_05111744", f"log-{seq_name}.txt")
        log_path = os.path.join("/bd_byt4090i0/users/omnimotion/dayu/d4sm3_logs/VT_improved", f"log-{seq_name}.txt")
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        with open(log_path, "w", buffering=1) as log_f, \
             contextlib.redirect_stdout(log_f), \
             contextlib.redirect_stderr(log_f):

            print(f"=== Sequence: {seq_name} ===")
            # Reset runtime state to avoid cross-sequence memory carry-over.
            save_vis_dir_vt = save_vis_dir if save_vis_dir is not None else "vis_results"
            tracker.reset_sequence_state(seq_name=seq_name, save_vis_dir=save_vis_dir_vt)
            torch.cuda.empty_cache()

            sequence = dataset[seq_name]
            objs = sequence.objects()
            objs_list = sorted(list(objs))
            sequence_len = len(sequence)

            if visualize:
                seq_save_dir = None
                seq_altern_save_dir = None
                if save_vis_dir is not None:
                    seq_save_dir = os.path.join(save_vis_dir, seq_name)
                    seq_altern_save_dir = os.path.join(seq_save_dir, "altern-masks")
                    seq_save_dir = os.path.join(seq_save_dir, "chosen-masks")
                visualizer = Visualizer(seq_len=sequence_len, save_dir=seq_save_dir, show_window=False)
                altern_visualizer = Visualizer(seq_len=sequence_len, save_dir=seq_altern_save_dir, show_window=False)

            per_frame_time = []
            pred_masks = []
            for ti in range(sequence_len):
                img_vis = sequence.frame(ti).image()
                image = Image.fromarray(img_vis)
                
                if ti == 0:
                    init_regions = []
                    for obj_id in objs_list:
                        # obj_id is a string, e.g., 'obj1', 'obj2', 'object', ...
                        init_region = sequence.object(obj_id)[ti]
                        """
                        if init_region.type == RegionType.MASK:
                            init_mask = init_region.rasterize((0, 0, image.width-1, image.height-1))
                            init_mask = (init_mask > 0.5).astype(np.uint8)
                            init_regions.append({'obj_id': obj_id, 'mask': init_mask})
                        elif init_region.type == RegionType.RECTANGLE:
                            bb_ = [init_region.x, init_region.y, init_region.width, init_region.height]
                            init_regions.append({'obj_id': obj_id, 'bbox': bb_})
                        else:
                            print('Error: Unknown init region type:', init_region.type)
                            exit(-1)
                        """
                        # 初始化时的区域判断
                        if isinstance(init_region, Mask):
                            # 注意：视具体 API 版本而定，rasterize 的参数可能是一个尺寸元组
                            init_mask = init_region.rasterize((0, 0, image.width-1, image.height-1))
                            init_mask = (init_mask > 0.5).astype(np.uint8)
                            init_regions.append({'obj_id': obj_id, 'mask': init_mask})

                        elif isinstance(init_region, Rectangle):
                            bb_ = [init_region.x, init_region.y, init_region.width, init_region.height]
                            init_regions.append({'obj_id': obj_id, 'bbox': bb_})

                        else:
                            # 打印真实的 Python 类名，方便万一报错时进行调试
                            print(f'Error: Unknown init region type: {type(init_region).__name__}')
                            exit(-1)
                    outputs = tracker.initialize(image, init_regions)
                    pred_masks = None
                else:
                    torch.cuda.synchronize()
                    t_ = time.time()
                    outputs = tracker.track(image)
                    torch.cuda.synchronize()
                    t_i = time.time() - t_
                    per_frame_time.append(t_i)
                    # print('%d: %.4f' % (ti, t_i))
                    pred_masks = outputs['masks']
                    alternative_masks = outputs['alternative_masks']
                    # pred_masks: list with n_obj elements, 
                    # where n_obj in the number of objects being tracked
                    # each element of the list: 
                    # numpy array (uint8) with zeros/ones

                if visualize:
                    if pred_masks is not None:
                        visualizer.visualize(img_vis.copy(), mask=pred_masks, frame_index=ti)
                        """
                        altern_visualizer.visualize(
                            img_vis.copy(),
                            mask=alternative_masks,
                            frame_index=ti,
                        )
                        """
                    else:
                        if 'mask' in init_regions[0]:
                            msks_ = [reg['mask'] for reg in init_regions]
                            visualizer.visualize(img_vis.copy(), mask=msks_, frame_index=ti)
                        elif 'bbox' in init_regions[0]:
                            bbxs_ = [reg['bbox'] for reg in init_regions]
                            visualizer.visualize(img_vis.copy(), bbox=bbxs_, frame_index=ti)
                        else:
                            print('Warning: cannot visualize intialization.')

            print('-----------------------')
            print('    %s: %d targets' % (seq_name, len(objs_list)))
            avg_time = sum(per_frame_time) / len(per_frame_time)
            avg_speed = 1 / avg_time
            print('    Average time: %.3f' % (avg_time))
            print('    Average speed: %.1f' % (avg_speed))
            torch.cuda.empty_cache()

    # Clear Hydra once after all sequences are processed.
    del tracker
    torch.cuda.empty_cache()
    hydra.core.global_hydra.GlobalHydra.instance().clear()


def main():
    parser = argparse.ArgumentParser(description='Visualize sequence.')
    parser.add_argument('--dataset', type=str, required=True, help='VOTS23 dataset path.')
    parser.add_argument('--sequence', type=str, default=None, help='Sequence name.')
    parser.add_argument('--checkpoint_dir', type=str, default=None, help='Checkpoint directory.')
    parser.add_argument('--visualize', action='store_true', help='Visualize.')
    parser.add_argument('--save_vis_dir', type=str, default='vis_results', help='Directory to save per-frame visualization images.')
    
    args = parser.parse_args()
    run_sequence(
        args.dataset,
        args.sequence,
        args.checkpoint_dir,
        args.visualize,
        args.save_vis_dir,
    )

if __name__ == "__main__":
    main()
