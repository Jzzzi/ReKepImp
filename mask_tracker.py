import multiprocessing as mp
import os
import sys

import torch
import numpy as np
import cv2

from cutie.inference.inference_core import InferenceCore
from cutie.utils.get_default_model import get_default_model
sys.path.append(os.path.dirname(__file__))
from utils.utils import fileter_masks_by_bounds, merge_masks

RESET = "\033[0m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"


def to_tensor(img:np.ndarray)->torch.Tensor:
    return torch.from_numpy(img.transpose(2, 0, 1)).float().contiguous()

class MaskTrackerProcess():
    def __init__(self, config, manual:bool=False):
        '''
        Set mp.set_start_method('spawn') in the main process.
        '''
        # process settings
        self._data_queue = mp.Queue()
        self._mask_queue = mp.Queue()
        self._stop_event = mp.Event()
        self._max_queue_size = config['max_queue_size']
        self._process = None

        # random seed
        np.random.seed(config['seed'])
        torch.manual_seed(config['seed'])
        torch.cuda.manual_seed(config['seed'])

        # mask tracker settings
        self._num_objects = config['num_objects']
        self._device = config['device']
        # workspace bounds
        bounds_min = np.array(config['bounds_min']).reshape(-1, 3)
        bounds_max = np.array(config['bounds_max']).reshape(-1, 3)
        self._bounds = np.concatenate([bounds_min, bounds_max], axis=0) # (2, 3)
        self._max_overlap_ratio = config['max_overlap_ratio'] # max overlap ratio for masks

        # SAM parameters
        self._manual = manual # whether manually select masks
        self._path = config['path'] # folder path of sam model
        self._max_mask_ratio = config['max_mask_ratio']
        self._min_mask_ratio = config['min_mask_ratio']
        self._points_per_side = config['points_per_side']
        self._crop_n_layers = config['crop_n_layers']
        self._crop_n_points_downscale_factor = config['crop_n_points_downscale_factor']
        
        self._segmented = False
        self._processor = None # Cutie processor
        self._objects = None # list of object ids, count from 1
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
        self._process.terminate()
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
                if self._manual:
                    # manually select masks
                    print(GREEN + f"[MaskTracker]: Manually selecting masks." + RESET)
                    cv2.namedWindow("Maskselection") # must be called before _get_sam_mask_manual, i don't know why, perhaps it's because of the multiprocessing
                    init_mask = self._get_sam_mask_manual(rgb)                   
                else:
                    # automatically select masks
                    print(GREEN + f"[MaskTracker]: Automatically selecting masks." + RESET)
                    init_mask = self._get_sam_mask_auto(rgb, points)
                

                self._init_cutie(rgb, init_mask)
                self._segmented = True

            else:
                # Use Cutie to track the masks
                self._track(data['rgb'])
        return

    @torch.inference_mode()
    @torch.amp.autocast('cuda')
    def _get_sam_mask_auto(self, rgb:np.ndarray, points:np.ndarray)->list:
        '''
        Return:
            masks: np.ndarray, [H, W], uint8, 0 for background, 1 for object 1, 2 for object 2, ...
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
        masks = [m['segmentation'] for m in masks]
        print(GREEN + f"[MaskTracker]: {len(masks)} masks generated." + RESET)
        length = len(masks)
        # filter masks by area ratio
        masks = [m for m in masks if m.sum()/(m.shape[0]*m.shape[1]) < self._max_mask_ratio]
        masks = [m for m in masks if m.sum()/(m.shape[0]*m.shape[1]) > self._min_mask_ratio]
        print(GREEN + f"[MaskTracker]: {length-len(masks)} masks filtered by ratio." + RESET)
        length = len(masks)
        # filter masks by bounds
        masks = fileter_masks_by_bounds(masks, points, self._bounds)
        print(GREEN + f"[MaskTracker]: {length-len(masks)} masks filtered by bounds." + RESET)
        length = len(masks)
        # filter masks by overlap ratio
        masks = merge_masks(masks, self._max_overlap_ratio)
        print(GREEN + f"[MaskTracker]: {length-len(masks)} masks merged." + RESET)
        print(GREEN + f"[MaskTracker]: {len(masks)} masks selected." + RESET)
        assert len(masks) > 0, "No masks generated."
        self._num_objects = min(len(masks), self._num_objects)
        masks = masks[:self._num_objects]
        for i, mask in enumerate(masks):
            print(GREEN + f"[MaskTracker]: Mask {i+1} area ratio: {mask.sum()/(mask.shape[0]*mask.shape[1])}" + RESET)

        init_mask = torch.zeros(rgb.shape[:2], dtype=torch.uint8).cuda()
        self._objects = [i + 1 for i in range(len(masks))] # count from 1
        for i in range(len(masks)):
            init_mask[masks[i]>0] = self._objects[i]
        torch.cuda.empty_cache()
        return init_mask

    @torch.inference_mode()
    @torch.amp.autocast('cuda')
    def _get_sam_mask_manual(self, rgb:np.ndarray)->np.ndarray:
        '''
        Get masks by manually selecting points
        '''
        from segment_anything import SamPredictor, sam_model_registry
        sam_checkpoint = os.path.join(self._path, "model", "sam_vit_h_4b8939.pth")
        model_type = "vit_h"
        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=self._device)
        predictor = SamPredictor(sam)
        predictor.set_image(rgb)

        masks = []
        bgr_image = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR) # for openCV display
        print(GREEN + f"[MaskTracker]: Select robot arm first." + RESET)
        while True:
            points = []
            lables = []
            print(GREEN + f"[MaskTracker]: Select points by left click, press s to confirm the points, press q to finish." + RESET)
            cv2.imshow("Maskselection", bgr_image)

            def _mouse_callback(event, x, y, flags, param):
                points, lables = param
                if event == cv2.EVENT_LBUTTONDOWN:
                    points.append([x, y])
                    lables.append(1)
                    cv2.circle(bgr_image, (x, y), 5, (0, 255, 0), -1)
                    cv2.imshow("Maskselection", bgr_image)

            cv2.setMouseCallback("Maskselection", _mouse_callback, param=[points, lables])

            key = cv2.waitKey(0)
            if key == ord('s'):
                assert len(points) > 0, "No points selected."
                points = np.array(points)
                lables = np.array(lables)
                mask, _, _ = predictor.predict(points, lables, multimask_output=False)
                mask = mask.reshape(rgb.shape[:2])
                masks.append(mask)
                # highlight the selected mask
                mask_rgb = np.zeros_like(rgb)
                mask_rgb[mask>0] = [0, 255, 0]
                bgr_image = cv2.addWeighted(bgr_image, 1, mask_rgb, 0.5, 0)
                print(GREEN + f"[MaskTracker]: Mask {len(masks) - 1} selected." + RESET)
            elif key == ord('q'):
                cv2.destroyAllWindows()
                break
        
        # merge masks
        self._num_objects = len(masks)
        self._objects = [i +1 for i in range(len(masks))] # count from 1
        init_mask = torch.zeros(rgb.shape[:2], dtype=torch.uint8).cuda()
        for i in range(len(masks)):
            init_mask[masks[i]>0] = self._objects[i]
        torch.cuda.empty_cache()
        return init_mask


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
        if self._mask_queue.qsize() > self._max_queue_size:
            self._mask_queue.get()
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
        if self._mask_queue.qsize() > self._max_queue_size:
            self._mask_queue.get()
