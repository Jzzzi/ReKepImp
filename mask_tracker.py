import multiprocessing as mp
import os
import sys

import torch
import numpy as np
import cv2

from cutie.inference.inference_core import InferenceCore
from cutie.utils.get_default_model import get_default_model
sys.path.append(os.path.dirname(__file__))
from utils.utils import fileter_masks_by_bounds

RESET = "\033[0m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"


def to_tensor(img:np.ndarray)->torch.Tensor:
    return torch.from_numpy(img.transpose(2, 0, 1)).float().contiguous()

class MaskTrackerProcess():
    def __init__(self, config):
        '''
        Set mp.set_start_method('spawn') in the main process.
        '''
        self._data_queue = mp.Queue()
        self._mask_queue = mp.Queue()
        self._stop_event = mp.Event()
        self.__sam_done_event = mp.Event()
        self._num_objects = config['num_objects']
        self._path = config['path']
        self._device = config['device']

        # SAM parameters
        self._max_mask_ratio = config['max_mask_ratio']
        self._min_mask_ratio = config['min_mask_ratio']
        self._points_per_side = config['points_per_side']
        self._crop_n_layers = config['crop_n_layers']
        self._crop_n_points_downscale_factor = config['crop_n_points_downscale_factor']
        
        bounds_min = np.array(config['bounds_min']).reshape(-1, 3)
        bounds_max = np.array(config['bounds_max']).reshape(-1, 3)
        self._bounds = np.concatenate([bounds_min, bounds_max], axis=0) # (2, 3)

        self._process = None
        self._segmented = False
        self._mask = None
        self._processor = None
        self._objects = None
        print(GREEN + f"[MaskTracker]: Mask tracker process initialized." + RESET)

    def start(self):
        print(GREEN + f"[MaskTracker]: Starting mask tracker process..." + RESET)
        self._process = mp.Process(target=self._run)
        self._process.start()

    def stop(self):
        print(GREEN + f"[MaskTracker]: Stopping mask tracker process..." + RESET)
        self._data_queue.close()
        self._mask_queue.close()
        self._stop_event.set()
        self._process.join()
        print(GREEN + f"[MaskTracker]: Mask tracker process stopped." + RESET)
    
    def send(self, data):
        '''
        Send data dict{
            'rgb': np.ndarray, [H, W, 3], uint8
            'points': np.ndarray, [H, W, 3], float32, points in world frame}
        '''
        self._data_queue.put(data)

    def get(self)->np.ndarray:
        '''
        Return the mask as np.ndarray, [H, W], uint8
        '''
        return self._mask_queue.get()
    
    def sam_done(self)->bool:
        return self.__sam_done_event.is_set()
    
    def _run(self):
        while not self._stop_event.is_set():
            try:
                data = self._data_queue.get(timeout=1)
            except:
                # print(YELLOW + f"[MaskTracker]: No data received in 1 seconds." + RESET)
                continue
            if not self._segmented:
                # Use the first frame to generate masks by SAM
                assert data is not None, "Data is None."
                rgb = data['rgb']
                points = data['points']
                masks = self._get_sam_mask(rgb)

                masks = [m['segmentation'] for m in masks]
                print(GREEN + f"[MaskTracker]: {len(masks)} masks generated." + RESET)
                masks = [m for m in masks if m.sum()/(m.shape[0]*m.shape[1]) < self._max_mask_ratio]
                masks = [m for m in masks if m.sum()/(m.shape[0]*m.shape[1]) > self._min_mask_ratio]
                masks = fileter_masks_by_bounds(masks, points, self._bounds)
                print(GREEN + f"[MaskTracker]: {len(masks)} masks selected." + RESET)
                self._num_objects = min(len(masks), self._num_objects)
                # DEBUG
                # print the area ratio of each mask
                masks = masks[:self._num_objects]
                for i, mask in enumerate(masks):
                    print(GREEN + f"[MaskTracker]: Mask {i} area ratio: {mask.sum()/(mask.shape[0]*mask.shape[1])}" + RESET)
                init_mask = torch.zeros(rgb.shape[:2], dtype=torch.uint8).cuda()
                self._objects = [i + 1 for i in range(len(masks))]
                for i in range(len(masks)):
                    init_mask[masks[i]>0] = self._objects[i]
                torch.cuda.empty_cache()
                self._init_cutie(rgb, init_mask)
                self._segmented = True

            else:
                # Use Cutie to track the masks
                self._track(data['rgb'])
        return

    @torch.inference_mode()
    @torch.amp.autocast('cuda')
    def _get_sam_mask(self, rgb:np.ndarray)->list:
        '''
        Mask generation returns a list over masks, where each mask is a dictionary containing various data about the mask. These keys are:
        segmentation : the mask
        area : the area of the mask in pixels
        bbox : the boundary box of the mask in XYWH format
        predicted_iou : the model's own prediction for the quality of the mask
        point_coords : the sampled input point that generated this mask
        stability_score : an additional measure of mask quality
        crop_box : the crop of the image used to generate this mask in XYWH format
        '''
        print(GREEN + f"[MaskTracker]: Initializing SAM model..." + RESET)
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        # sam model initialization
        sam_checkpoint = os.path.join(self._path, "model", "sam_vit_h_4b8939.pth")
        model_type = "vit_h"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=self._device)
        mask_generator = SamAutomaticMaskGenerator(sam,
                                                    points_per_side=self._points_per_side,
                                                    # pred_iou_thresh=0.85,
                                                    # stability_score_thresh=0.85,
                                                    crop_n_layers=self._crop_n_layers,
                                                    crop_n_points_downscale_factor=self._crop_n_points_downscale_factor,
                                                    # min_mask_region_area=50
                                                    )

        print(GREEN + f"[MaskTracker]: Generating masks..." + RESET)
        masks = mask_generator.generate(rgb)
        # sorted by te predicted_iou
        masks = sorted(masks, key=lambda x: x['predicted_iou'], reverse=True)
        self.__sam_done_event.set()
        return masks

    @torch.inference_mode()
    @torch.amp.autocast('cuda')
    def _init_cutie(self, first_frame, init_mask):
        print(GREEN + f"[MaskTracker]: Initializing Cutie tracker..." + RESET)
        cutie = get_default_model()
        self._processor = InferenceCore(cutie, cfg=cutie.cfg)
        self._processor.max_internal_size = 480
        output_prob = self._processor.step(to_tensor(first_frame).cuda().float()/255, init_mask, objects=self._objects)
        mask = self._processor.output_prob_to_mask(output_prob)
        mask = mask.cpu().numpy()
        self._mask_queue.put(mask)
        print(GREEN + f"[MaskTracker]: Cutie tracker initialized." + RESET)

    @torch.inference_mode()
    @torch.amp.autocast('cuda')
    def _track(self, rgb):
        # Track the following frames
        rgb_tensor = to_tensor(rgb).cuda().float()/255

        assert self._processor is not None, "Cutie processor is None."

        output_prob = self._processor.step(rgb_tensor)
        mask = self._processor.output_prob_to_mask(output_prob)
        mask = mask.cpu().numpy()
        self._mask_queue.put(mask)
