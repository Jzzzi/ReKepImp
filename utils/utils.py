import sys
import os
import yaml


import numpy as np
import torch
import cv2
from scipy.spatial.transform import Rotation as R

def get_cam_points(depth:np.ndarray, instrinsics:np.ndarray)->np.ndarray:
    '''
    depth: [H, W], depth image in z16 format
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
    Transforms depth map points from camera coordinates to world coordinates.

    Args:
        depth (np.ndarray): A 2D array representing the depth image in z16 format.
        instrinsics (np.ndarray): A 3x3 array representing the camera intrinsic parameters.
        extrinsics (np.ndarray): A 4x4 array representing the camera-to-world transformation matrix (extrinsics).

    Returns:
        np.ndarray: A 3D array of shape (h, w, 3) representing the transformed points in world coordinates.
    '''
    points = get_cam_points(depth, instrinsics)
    h, w = depth.shape
    points = points.reshape(-1, 3)
    points = points @ extrinsics[:3, :3].T + extrinsics[:3, 3]
    return points.reshape(h, w, 3)

def filter_points_by_bounds(points, bounds_min, bounds_max, strict=True):
    """
    Filter points by taking only points within workspace bounds.
    """
    assert points.shape[1] == 3, "points must be (N, 3)"
    bounds_min = bounds_min.copy()
    bounds_max = bounds_max.copy()
    if not strict:
        bounds_min[:2] = bounds_min[:2] - 0.1 * (bounds_max[:2] - bounds_min[:2])
        bounds_max[:2] = bounds_max[:2] + 0.1 * (bounds_max[:2] - bounds_min[:2])
        bounds_min[2] = bounds_min[2] - 0.1 * (bounds_max[2] - bounds_min[2])
    within_bounds_mask = (
        (points[:, 0] >= bounds_min[0])
        & (points[:, 0] <= bounds_max[0])
        & (points[:, 1] >= bounds_min[1])
        & (points[:, 1] <= bounds_max[1])
        & (points[:, 2] >= bounds_min[2])
        & (points[:, 2] <= bounds_max[2])
    )
    return within_bounds_mask

import numpy as np

def quat2mat(quat:np.array)->np.array:
    '''
    Convert quaternion to rotation matrix

    Args:
        quat (np.array): A 4-element array representing the quaternion, q = [x, y, z, w].

    Returns:
        np.array: A 3x3 array representing the rotation matrix.
    '''
    return R.from_quat(quat).as_matrix()


def mat2quat(matrix:np.array)->np.array:
    '''
    Convert rotation matrix to quaternion

    Args:
        matrix (np.array): A 3x3 array representing the rotation matrix.

    Returns:
        np.array: A 4-element array representing the quaternion, q = [x, y, z, w].
    '''
    return R.from_matrix(matrix).as_quat()

def get_config(path="/home/liujk/ReKepImp/config/config.yaml") -> dict:
    '''
    Load config file

    Args:
        path (str): Path to the config file.

    Returns:
        dict: A dictionary containing the config file.
    '''
    with open(path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config