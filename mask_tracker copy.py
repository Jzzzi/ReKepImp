import multiprocessing as mp
import os
import sys

import torch
import numpy as np
import cv2

sys.path.append(os.path.join(os.path.dirname(__file__), "cutie"))

from cutie.inference.inference_core import InferenceCore
from cutie.utils.get_default_model import get_default_model

def to_tensor(img:np.ndarray)->torch.Tensor:
    return torch.from_numpy(img.transpose(2, 0, 1)).float().contiguous()

class MaskTrackerProcess():
    def __init__(self, data_queue, mask_queue, stop_event, seg_over_event):
        self.data_queue = data_queue
        self.mask_queue = mask_queue
        self.stop_event = stop_event
        self.seg_over_event = seg_over_event
        self.process = None
        self.segmented = False
        self.mask = None
        self.processor = None
        self.objects = None

    def start(self):
        self.process = mp.Process(target=self.run)
        self.process.start()

    def stop(self):
        self.stop_event.set()
        self.process.join()
    
    def run(self):
        print("Mask tracker process started.")
        while not self.stop_event.is_set():
            print("Waiting for data...")
            data = self.data_queue.get()
            print("Data received.")
            if not self.segmented:
                # Use the first frame to generate masks by SAM
                print("Segmenting the first frame...")
                assert data is not None, "Data is None."
                bgr = data['color']
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                masks = self.get_sam_mask(rgb)
                # Only track the top 10 masks
                masks = [m['segmentation'] for m in masks[:10]]
                self.mask = torch.zeros(rgb.shape[:2], dtype=torch.uint8).cuda()
                self.objects = [i + 1 for i in range(len(masks))]
                for i in range(len(masks)):
                    self.mask[masks[i]>0] = i + 1
                
                self.init_cutie(rgb)
                self.mask_queue.put(self.mask)
                self.segmented = True
                self.seg_over_event.set()

            else:
                # Use Cutie to track the masks
                print("Tracking the mask...")
                self.track(data)

    @torch.inference_mode()
    @torch.amp.autocast('cuda')
    def get_sam_mask(self, rgb:np.ndarray)->list:
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
        sam_checkpoint = os.path.join(os.path.dirname(os.path.dirname(__file__)), "model", "sam_vit_h_4b8939.pth")
        model_type = "vit_h"
        device = "cuda"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        mask_generator = SamAutomaticMaskGenerator(sam)

        print('Generating masks...')
        masks = mask_generator.generate(rgb)
        print(f'Total masks: {len(masks)}')
        # sorted by te predicted_iou
        print('Sorting masks...')
        masks = sorted(masks, key=lambda x: x['predicted_iou'], reverse=True)
        return masks

    @torch.inference_mode()
    @torch.amp.autocast('cuda')
    def init_cutie(self, first_frame):
        print("Initializing Cutie tracker...")
        cutie = get_default_model()
        self.processor = InferenceCore(cutie, cfg=cutie.cfg)
        self.processor.max_internal_size = 480
        output_prob = self.processor.step(to_tensor(first_frame).cuda().float(), self.mask, objects=self.objects)
        print("Cutie tracker initialized.")

    @torch.inference_mode()
    @torch.amp.autocast('cuda')
    def track(self, data):
        # Track the following frames
        bgr = data['color']
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb_tensor = to_tensor(rgb).cuda().float()

        if self.processor is not None:
            print("Tracking mask in current frame...")
            output_prob = self.processor.step(rgb_tensor)
            mask = self.processor.output_prob_to_mask(output_prob)

            # 将生成的mask发送到mask_queue
            self.mask_queue.put(mask)
            print("Mask tracked and sent to queue.")
