import sys
import os

import numpy as np
import torch
import cv2

def get_cam_points(depth:np.ndarray, instrinsics:np.ndarray)->np.ndarray:
    '''
    points, [H, W, 3], 3D points in camera coordinate
    '''
    h, w = depth.shape
    # conver z16 to float in unit of meters
    depth = depth.astype(float) * 0.001
    fx, fy, cx, cy = instrinsics[0], instrinsics[4], instrinsics[2], instrinsics[5]
    direction_x, direction_y = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')
    direction_x = (direction_x.astype(float) - cx) / fx
    direction_y = (direction_y.astype(float) - cy) / fy
    direction = np.stack([direction_x, direction_y, np.ones_like(direction_x)], axis=-1)
    points = direction * depth[:, :, np.newaxis]
    return points

def get_points(depth:np.ndarray, instrinsics:np.ndarray, extrinsics:np.ndarray)->np.ndarray:
    '''
    extrinsics, [4, 4], c2w matrix    
    '''
    points = get_cam_points(depth, instrinsics)
    h, w = depth.shape
    points = points.reshape(-1, 3)
    points = points @ extrinsics[:3, :3].T + extrinsics[:3, 3]
    return points.reshape(h, w, 3)

def get_sam_mask(rgb:np.ndarray)->list:
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

