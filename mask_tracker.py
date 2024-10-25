import multiprocessing as mp
import os
import sys

import torch
import numpy as np
import cv2

from cutie.inference.inference_core import InferenceCore
from cutie.utils.get_default_model import get_default_model

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
        Send data as np.ndarray, [H, W, 3], BGR
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
        print("Mask tracker process started.")
        while not self._stop_event.is_set():
            try:
                data = self._data_queue.get(timeout=1)
            except:
                # print(YELLOW + f"[MaskTracker]: No data received in 1 seconds." + RESET)
                continue
            if not self._segmented:
                # Use the first frame to generate masks by SAM
                assert data is not None, "Data is None."
                bgr = data
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                masks = self._get_sam_mask(rgb)
                self._num_objects = min(self._num_objects, len(masks))
                masks = [m['segmentation'] for m in masks[:self._num_objects]]
                self._mask = torch.zeros(rgb.shape[:2], dtype=torch.uint8).cuda()
                self._objects = [i + 1 for i in range(len(masks))]
                for i in range(len(masks)):
                    self._mask[masks[i]>0] = i + 1
                
                self._init_cutie(rgb)
                self._mask_queue.put(self._mask.cpu().numpy())
                self._segmented = True

            else:
                # Use Cutie to track the masks
                self._track(data)
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
        print('Initializing the mask generator...')
        from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
        # sam model initialization
        sam_checkpoint = os.path.join(self._path, "model", "sam_vit_h_4b8939.pth")
        model_type = "vit_h"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=self._device)
        mask_generator = SamAutomaticMaskGenerator(sam)

        print('Generating masks...')
        masks = mask_generator.generate(rgb)
        # sorted by te predicted_iou
        print('Sorting masks...')
        masks = sorted(masks, key=lambda x: x['predicted_iou'], reverse=True)
        self.__sam_done_event.set()
        return masks

    @torch.inference_mode()
    @torch.amp.autocast('cuda')
    def _init_cutie(self, first_frame):
        print("Initializing Cutie tracker...")
        cutie = get_default_model()
        self._processor = InferenceCore(cutie, cfg=cutie.cfg)
        self._processor.max_internal_size = 480
        output_prob = self._processor.step(to_tensor(first_frame).cuda().float()/255, self._mask, objects=self._objects)
        print("Cutie tracker initialized.")

    @torch.inference_mode()
    @torch.amp.autocast('cuda')
    def _track(self, data):
        # Track the following frames
        bgr = data
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb_tensor = to_tensor(rgb).cuda().float()/255

        assert self._processor is not None, "Cutie processor is None."

        output_prob = self._processor.step(rgb_tensor)
        mask = self._processor.output_prob_to_mask(output_prob)
        mask = mask.cpu().numpy()
        self._mask_queue.put(mask)
