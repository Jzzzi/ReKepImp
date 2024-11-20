import sys
import os
import yaml


import numpy as np
import torch
import cv2
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import RotationSpline
import scipy.interpolate as interpolate
import open3d as o3d
from numba import njit

RESET = "\033[0m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"

def get_cam_points(depth:np.ndarray, instrinsics:np.ndarray, mask:np.ndarray=None)->np.ndarray:
    '''
    depth: [H, W], depth image in z16 format
    points, [H, W, 3], 3D points in camera coordinate
    '''
    h, w = depth.shape
    # conver z16 to float in unit of meters
    depth = depth.astype(float) * 0.001
    if mask is not None:
        binary_mask = (mask > 0)
        depth[~binary_mask] = 0
    instrinsics = instrinsics.reshape(-1)
    assert instrinsics.shape == (9,), instrinsics.shape
    fx, fy, cx, cy = instrinsics[0], instrinsics[4], instrinsics[2], instrinsics[5]
    direction_x, direction_y = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')
    direction_x = (direction_x.astype(float) - cx) / fx
    direction_y = (direction_y.astype(float) - cy) / fy
    direction = np.stack([direction_x, direction_y, np.ones_like(direction_x)], axis=-1)
    points = direction * depth[:, :, np.newaxis]
    return points

def get_points(depth:np.ndarray, instrinsics:np.ndarray, extrinsics:np.ndarray, mask:np.ndarray=None)->np.ndarray:
    '''
    Transforms depth map points from camera coordinates to world coordinates.

    Args:
        depth (np.ndarray): A 2D array representing the depth image in z16 format.
        instrinsics (np.ndarray): A 3x3 array representing the camera intrinsic parameters.
        extrinsics (np.ndarray): A 4x4 array representing the camera-to-world transformation matrix (extrinsics).

    Returns:
        np.ndarray: A 3D array of shape (h, w, 3) representing the transformed points in world coordinates.
    '''
    points = get_cam_points(depth, instrinsics, mask)
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

def unnormalize_vars(normalized_vars, bounds):
    """
    Given 1D variables in [-1, 1] and original bounds, denormalize the variables to the original range.
    """
    vars = np.empty_like(normalized_vars)
    for i, (b_min, b_max) in enumerate(bounds):
        vars[i] = (normalized_vars[i] + 1) / 2 * (b_max - b_min) + b_min
    return vars

def normalize_vars(vars, bounds):
    """
    Given 1D variables and bounds, normalize the variables to [-1, 1] range.
    """
    normalized_vars = np.empty_like(vars)
    for i, (b_min, b_max) in enumerate(bounds):
        normalized_vars[i] = (vars[i] - b_min) / (b_max - b_min) * 2 - 1
    return normalized_vars

def pose2mat(pose):
    """
    Converts pose to homogeneous matrix.

    Args:
        pose (2-tuple): a (pos, orn) tuple where pos is vec3 float cartesian, and
            orn is vec4 float quaternion.

    Returns:
        np.array: 4x4 homogeneous matrix
    """
    homo_pose_mat = np.zeros((4, 4), dtype=pose[0].dtype)
    homo_pose_mat[:3, :3] = quat2mat(pose[1])
    homo_pose_mat[:3, 3] = np.array(pose[0], dtype=pose[0].dtype)
    homo_pose_mat[3, 3] = 1.0
    return homo_pose_mat

def euler2quat(euler):
    """
    Converts euler angles into quaternion form

    Args:
        euler (np.array): (r,p,y) angles

    Returns:
        np.array: (x,y,z,w) float quaternion angles

    Raises:
        AssertionError: [Invalid input shape]
    """
    return R.from_euler("xyz", euler).as_quat()

def quat2euler(quat):
    """
    Converts euler angles into quaternion form

    Args:
        quat (np.array): (x,y,z,w) float quaternion angles

    Returns:
        np.array: (r,p,y) angles

    Raises:
        AssertionError: [Invalid input shape]
    """
    return R.from_quat(quat).as_euler("xyz")

@njit(cache=True, fastmath=True)
def angle_between_rotmat(P, Q):
    R = np.dot(P, Q.T)
    cos_theta = (np.trace(R)-1)/2
    if cos_theta > 1:
        cos_theta = 1
    elif cos_theta < -1:
        cos_theta = -1
    return np.arccos(cos_theta)

@njit(cache=True, fastmath=True)
def consistency(poses_a, poses_b, rot_weight=0.5):
    assert poses_a.shape[1:] == (4, 4) and poses_b.shape[1:] == (4, 4), 'poses must be of shape (N, 4, 4)'
    min_distances = np.zeros(len(poses_a), dtype=np.float64)
    for i in range(len(poses_a)):
        min_distance = 9999999
        a = poses_a[i]
        for j in range(len(poses_b)):
            b = poses_b[j]
            pos_distance = np.linalg.norm(a[:3, 3] - b[:3, 3])
            rot_distance = angle_between_rotmat(a[:3, :3], b[:3, :3])
            distance = pos_distance + rot_distance * rot_weight
            min_distance = min(min_distance, distance)
        min_distances[i] = min_distance
    return np.mean(min_distances)

def transform_keypoints(transform, keypoints, movable_mask):
    '''
    Transform movable keypoints using a 4x4 transformation matrix.
    '''
    assert transform.shape == (4, 4)
    transformed_keypoints = keypoints.copy()
    if movable_mask.sum() > 0:
        transformed_keypoints[movable_mask] = np.dot(keypoints[movable_mask], transform[:3, :3].T) + transform[:3, 3]
    return transformed_keypoints

def farthest_point_sampling(pc, num_points):
    """
    Given a point cloud, sample num_points points that are the farthest apart.
    Use o3d farthest point sampling.
    """
    assert pc.ndim == 2 and pc.shape[1] == 3, "pc must be a (N, 3) numpy array"
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    downpcd_farthest = pcd.farthest_point_down_sample(num_points)
    return np.asarray(downpcd_farthest.points)

@njit(cache=True, fastmath=True)
def batch_transform_points(points, transforms):
    """
    Apply multiple of transformation to point cloud, return results of individual transformations.
    Args:
        points: point cloud (N, 3).
        transforms: M 4x4 transformations (M, 4, 4).
    Returns:
        np.array: point clouds (M, N, 3).
    """
    assert transforms.shape[1:] == (4, 4), 'transforms must be of shape (M, 4, 4)'
    transformed_points = np.zeros((transforms.shape[0], points.shape[0], 3))
    for i in range(transforms.shape[0]):
        pos, R = transforms[i, :3, 3], transforms[i, :3, :3]
        transformed_points[i] = np.dot(points, R.T) + pos
    return transformed_points

def calculate_collision_cost(poses, sdf_func, collision_points, threshold):
    assert poses.shape[1:] == (4, 4)
    transformed_pcs = batch_transform_points(collision_points, poses)
    transformed_pcs_flatten = transformed_pcs.reshape(-1, 3)  # [num_poses * num_points, 3]
    signed_distance = sdf_func(transformed_pcs_flatten) + threshold  # [num_poses * num_points]
    signed_distance = signed_distance.reshape(-1, collision_points.shape[0])  # [num_poses, num_points]
    non_zero_mask = signed_distance > 0
    collision_cost = np.sum(signed_distance[non_zero_mask])
    return collision_cost

def euler2mat(euler):
    """
    Converts euler angles into rotation matrix form

    Args:
        euler (np.array): (r,p,y) angles

    Returns:
        np.array: 3x3 rotation matrix

    Raises:
        AssertionError: [Invalid input shape]
    """

    euler = np.asarray(euler, dtype=np.float64)
    assert euler.shape[-1] == 3, "Invalid shaped euler {}".format(euler)

    return R.from_euler("xyz", euler).as_matrix()


def convert_pose_euler2mat(poses_euler):
    """
    Convert poses from euler to mat format.
    Args:
    - poses_euler (np.ndarray): [N, 6]
    Returns:
    - poses_mat (np.ndarray): [N, 4, 4]
    """
    batched = poses_euler.ndim == 2
    if not batched:
        poses_euler = poses_euler[None]
    poses_mat = np.eye(4)
    poses_mat = np.tile(poses_mat, (len(poses_euler), 1, 1))
    poses_mat[:, :3, 3] = poses_euler[:, :3]
    for i in range(len(poses_euler)):
        poses_mat[i, :3, :3] = euler2mat(poses_euler[i, 3:])
    if not batched:
        poses_mat = poses_mat[0]
    return poses_mat

def convert_pose_mat2quat(poses_mat):
    """
    Convert poses from mat to quat xyzw format.
    Args:
    - poses_mat (np.ndarray): [N, 4, 4]
    Returns:
    - poses_quat (np.ndarray): [N, 7], [x, y, z, x, y, z, w]
    """
    batched = poses_mat.ndim == 3
    if not batched:
        poses_mat = poses_mat[None]
    poses_quat = np.empty((len(poses_mat), 7))
    poses_quat[:, :3] = poses_mat[:, :3, 3]
    for i in range(len(poses_mat)):
        poses_quat[i, 3:] = mat2quat(poses_mat[i, :3, :3])
    if not batched:
        poses_quat = poses_quat[0]
    return poses_quat

@njit(cache=True, fastmath=True)
def quat_slerp_jitted(quat0, quat1, fraction, shortestpath=True):
    """
    Return spherical linear interpolation between two quaternions.
    (adapted from deoxys)
    Args:
        quat0 (np.array): (x,y,z,w) quaternion startpoint
        quat1 (np.array): (x,y,z,w) quaternion endpoint
        fraction (float): fraction of interpolation to calculate
        shortestpath (bool): If True, will calculate the shortest path

    Returns:
        np.array: (x,y,z,w) quaternion distance
    """
    EPS = 1e-8
    q0 = quat0 / np.linalg.norm(quat0)
    q1 = quat1 / np.linalg.norm(quat1)
    if fraction == 0.0:
        return q0
    elif fraction == 1.0:
        return q1
    d = np.dot(q0, q1)
    if np.abs(np.abs(d) - 1.0) < EPS:
        return q0
    if shortestpath and d < 0.0:
        # invert rotation
        d = -d
        q1 *= -1.0
    if d < -1.0:
        d = -1.0
    elif d > 1.0:
        d = 1.0
    angle = np.arccos(d)
    if np.abs(angle) < EPS:
        return q0
    isin = 1.0 / np.sin(angle)
    q0 *= np.sin((1.0 - fraction) * angle) * isin
    q1 *= np.sin(fraction * angle) * isin
    q0 += q1
    return q0

@njit(cache=True, fastmath=True)
def get_samples_jitted(control_points_homo, control_points_quat, opt_interpolate_pos_step_size, opt_interpolate_rot_step_size):
    assert control_points_homo.shape[1:] == (4, 4)
    # calculate number of samples per segment
    num_samples_per_segment = np.empty(len(control_points_homo) - 1, dtype=np.int64)
    for i in range(len(control_points_homo) - 1):
        start_pos = control_points_homo[i, :3, 3]
        start_rotmat = control_points_homo[i, :3, :3]
        end_pos = control_points_homo[i+1, :3, 3]
        end_rotmat = control_points_homo[i+1, :3, :3]
        pos_diff = np.linalg.norm(start_pos - end_pos)
        rot_diff = angle_between_rotmat(start_rotmat, end_rotmat)
        pos_num_steps = np.ceil(pos_diff / opt_interpolate_pos_step_size)
        rot_num_steps = np.ceil(rot_diff / opt_interpolate_rot_step_size)
        num_path_poses = int(max(pos_num_steps, rot_num_steps))
        num_path_poses = max(num_path_poses, 2)  # at least 2 poses, start and end
        num_samples_per_segment[i] = num_path_poses
    # fill in samples
    num_samples = num_samples_per_segment.sum()
    samples_7 = np.empty((num_samples, 7))
    sample_idx = 0
    for i in range(len(control_points_quat) - 1):
        start_pos, start_xyzw = control_points_quat[i, :3], control_points_quat[i, 3:]
        end_pos, end_xyzw = control_points_quat[i+1, :3], control_points_quat[i+1, 3:]
        # using proper quaternion slerp interpolation
        poses_7 = np.empty((num_samples_per_segment[i], 7))
        for j in range(num_samples_per_segment[i]):
            alpha = j / (num_samples_per_segment[i] - 1)
            pos = start_pos * (1 - alpha) + end_pos * alpha
            blended_xyzw = quat_slerp_jitted(start_xyzw, end_xyzw, alpha)
            pose_7 = np.empty(7)
            pose_7[:3] = pos
            pose_7[3:] = blended_xyzw
            poses_7[j] = pose_7
        samples_7[sample_idx:sample_idx+num_samples_per_segment[i]] = poses_7
        sample_idx += num_samples_per_segment[i]
    assert num_samples >= 2, f'num_samples: {num_samples}'
    return samples_7, num_samples

def convert_pose_quat2mat(poses_quat):
    """
    Convert poses from quat xyzw to mat format.
    Args:
    - poses_quat (np.ndarray): [N, 7], [x, y, z, x, y, z, w]
    Returns:
    - poses_mat (np.ndarray): [N, 4, 4]
    """
    batched = poses_quat.ndim == 2
    if not batched:
        poses_quat = poses_quat[None]
    poses_mat = np.eye(4)
    poses_mat = np.tile(poses_mat, (len(poses_quat), 1, 1))
    poses_mat[:, :3, 3] = poses_quat[:, :3]
    for i in range(len(poses_quat)):
        poses_mat[i, :3, :3] = quat2mat(poses_quat[i, 3:])
    if not batched:
        poses_mat = poses_mat[0]
    return poses_mat

@njit(cache=True, fastmath=True)
def path_length(samples_homo):
    assert samples_homo.shape[1:] == (4, 4), 'samples_homo must be of shape (N, 4, 4)'
    pos_length = 0
    rot_length = 0
    for i in range(len(samples_homo) - 1):
        pos_length += np.linalg.norm(samples_homo[i, :3, 3] - samples_homo[i+1, :3, 3])
        rot_length += angle_between_rotmat(samples_homo[i, :3, :3], samples_homo[i+1, :3, :3])
    return pos_length, rot_length

def get_linear_interpolation_steps(start_pose, end_pose, pos_step_size, rot_step_size):
    """
    Given start and end pose, calculate the number of steps to interpolate between them.
    Args:
        start_pose: [6] position + euler or [4, 4] pose or [7] position + quat
        end_pose: [6] position + euler or [4, 4] pose or [7] position + quat
        pos_step_size: position step size
        rot_step_size: rotation step size
    Returns:
        num_path_poses: number of poses to interpolate
    """
    if start_pose.shape == (6,) and end_pose.shape == (6,):
        start_pos, start_euler = start_pose[:3], start_pose[3:]
        end_pos, end_euler = end_pose[:3], end_pose[3:]
        start_rotmat = euler2mat(start_euler)
        end_rotmat = euler2mat(end_euler)
    elif start_pose.shape == (4, 4) and end_pose.shape == (4, 4):
        start_pos = start_pose[:3, 3]
        start_rotmat = start_pose[:3, :3]
        end_pos = end_pose[:3, 3]
        end_rotmat = end_pose[:3, :3]
    elif start_pose.shape == (7,) and end_pose.shape == (7,):
        start_pos, start_quat = start_pose[:3], start_pose[3:]
        start_rotmat = quat2mat(start_quat)
        end_pos, end_quat = end_pose[:3], end_pose[3:]
        end_rotmat = quat2mat(end_quat)
    else:
        raise ValueError('start_pose and end_pose not recognized')
    pos_diff = np.linalg.norm(start_pos - end_pos)
    rot_diff = angle_between_rotmat(start_rotmat, end_rotmat)
    pos_num_steps = np.ceil(pos_diff / pos_step_size)
    rot_num_steps = np.ceil(rot_diff / rot_step_size)
    num_path_poses = int(max(pos_num_steps, rot_num_steps))
    num_path_poses = max(num_path_poses, 2)  # at least start and end poses
    return num_path_poses

def linear_interpolate_poses(start_pose, end_pose, num_poses):
    """
    Interpolate between start and end pose.
    """
    assert num_poses >= 2, 'num_poses must be at least 2'
    if start_pose.shape == (6,) and end_pose.shape == (6,):
        start_pos, start_euler = start_pose[:3], start_pose[3:]
        end_pos, end_euler = end_pose[:3], end_pose[3:]
        start_rotmat = euler2mat(start_euler)
        end_rotmat = euler2mat(end_euler)
    elif start_pose.shape == (4, 4) and end_pose.shape == (4, 4):
        start_pos = start_pose[:3, 3]
        start_rotmat = start_pose[:3, :3]
        end_pos = end_pose[:3, 3]
        end_rotmat = end_pose[:3, :3]
    elif start_pose.shape == (7,) and end_pose.shape == (7,):
        start_pos, start_quat = start_pose[:3], start_pose[3:]
        start_rotmat = quat2mat(start_quat)
        end_pos, end_quat = end_pose[:3], end_pose[3:]
        end_rotmat = quat2mat(end_quat)
    else:
        raise ValueError('start_pose and end_pose not recognized')
    slerp = Slerp([0, 1], R.from_matrix([start_rotmat, end_rotmat]))
    poses = []
    for i in range(num_poses):
        alpha = i / (num_poses - 1)
        pos = start_pos * (1 - alpha) + end_pos * alpha
        rotmat = slerp(alpha).as_matrix()
        if start_pose.shape == (6,):
            euler = mat2euler(rotmat)
            poses.append(np.concatenate([pos, euler]))
        elif start_pose.shape == (4, 4):
            pose = np.eye(4)
            pose[:3, :3] = rotmat
            pose[:3, 3] = pos
            poses.append(pose)
        elif start_pose.shape == (7,):
            quat = mat2quat(rotmat)
            pose = np.concatenate([pos, quat])
            poses.append(pose)
    return np.array(poses)

def convert_pose_euler2quat(poses_euler):
    """
    Convert poses from euler to quat xyzw format.
    Args:
    - poses_euler (np.ndarray): [N, 6]
    Returns:
    - poses_quat (np.ndarray): [N, 7], [x, y, z, x, y, z, w]
    """
    batched = poses_euler.ndim == 2
    if not batched:
        poses_euler = poses_euler[None]
    poses_quat = np.empty((len(poses_euler), 7))
    poses_quat[:, :3] = poses_euler[:, :3]
    for i in range(len(poses_euler)):
        poses_quat[i, 3:] = euler2quat(poses_euler[i, 3:])
    if not batched:
        poses_quat = poses_quat[0]
    return poses_quat

def mat2euler(rmat):
    """
    Converts given rotation matrix to euler angles in radian.

    Args:
        rmat (np.array): 3x3 rotation matrix

    Returns:
        np.array: (r,p,y) converted euler angles in radian vec3 float
    """
    M = np.array(rmat, dtype=rmat.dtype, copy=False)[:3, :3]
    return R.from_matrix(M).as_euler("xyz")

def fileter_masks_by_bounds(masks, points, bounds):
    '''
    Args:
        masks: list of np.ndarray of [H, W], uint8, masks
        points: np.ndarray, [N, 3], float32, points in world frame
        bounds: np.ndarray, [2, 3], float32, bounds in world frame
    '''
    filtered_masks = []
    for mask in masks:
        bool_mask = (mask > 0)
        mask_points = points[bool_mask]
        center_point = mask_points.mean(axis=0) # (3)
        if (center_point >= bounds[0]).all() and (center_point <= bounds[1]).all():
            filtered_masks.append(mask)
    return filtered_masks

def get_callable_grasping_cost_fn(env):
    def get_grasping_cost(keypoint_idx):
        keypoint_object = env.get_object_by_keypoint(keypoint_idx)
        # print(keypoint_idx)
        grasped = env.is_grasping(candidate_obj=keypoint_object)
        # print(grasped)
        return -grasped + 1 # return 0 if grasping an object, 1 if not grasping any object
    return get_grasping_cost

def merge_dicts(dicts):
    return {
        k : v 
        for d in dicts
        for k, v in d.items()
    }

def exec_safe(code_str, gvars=None, lvars=None):
    banned_phrases = ['import', '__']
    for phrase in banned_phrases:
        assert phrase not in code_str
  
    if gvars is None:
        gvars = {}
    if lvars is None:
        lvars = {}
    empty_fn = lambda *args, **kwargs: None
    custom_gvars = merge_dicts([
        gvars,
        {'exec': empty_fn, 'eval': empty_fn}
    ])
    try:
        exec(code_str, custom_gvars, lvars)
    except Exception as e:
        print(f'Error executing code:\n{code_str}')
        raise e

def load_functions_from_txt(txt_path, get_grasping_cost_fn):
    if txt_path is None:
        return []
    # load txt file
    with open(txt_path, 'r') as f:
        functions_text = f.read()
    # execute functions
    gvars_dict = {
        'np': np,
        'get_grasping_cost_by_keypoint_idx': get_grasping_cost_fn,
    }  # external library APIs
    lvars_dict = dict()
    exec_safe(functions_text, gvars=gvars_dict, lvars=lvars_dict)
    return list(lvars_dict.values())

def print_opt_debug_dict(debug_dict):
    print('\n' + '#' * 40)
    print(f'# Optimization debug info:')
    max_key_length = max(len(str(k)) for k in debug_dict.keys())
    for k, v in debug_dict.items():
        if isinstance(v, int) or isinstance(v, float):
            print(f'# {k:<{max_key_length}}: {v:.05f}')
        elif isinstance(v, list) and all(isinstance(x, int) or isinstance(x, float) for x in v):
            print(f'# {k:<{max_key_length}}: {np.array(v).round(5)}')
        else:
            print(f'# {k:<{max_key_length}}: {v}')
    print('#' * 40 + '\n')

def fit_b_spline(control_points):
    # determine appropriate k
    k = min(3, control_points.shape[0]-1)
    spline = interpolate.splprep(control_points.T, s=0, k=k)
    return spline

def sample_from_spline(spline, num_samples):
    sample_points = np.linspace(0, 1, num_samples)
    if isinstance(spline, RotationSpline):
        samples = spline(sample_points).as_matrix()  # [num_samples, 3, 3]
    else:
        assert isinstance(spline, tuple) and len(spline) == 2, 'spline must be a tuple of (tck, u)'
        tck, u = spline
        samples = interpolate.splev(np.linspace(0, 1, num_samples), tck)  # [spline_dim, num_samples]
        samples = np.array(samples).T  # [num_samples, spline_dim]
    return samples

def spline_interpolate_poses(control_points, num_steps):
    """
    Interpolate between through the control points using spline interpolation.
    1. Fit a b-spline through the positional terms of the control points.
    2. Fit a RotationSpline through the rotational terms of the control points.
    3. Sample the b-spline and RotationSpline at num_steps.

    Args:
        control_points: [N, 6] position + euler or [N, 4, 4] pose or [N, 7] position + quat
        num_steps: number of poses to interpolate
    Returns:
        poses: [num_steps, 6] position + euler or [num_steps, 4, 4] pose or [num_steps, 7] position + quat
    """
    assert num_steps >= 2, 'num_steps must be at least 2'
    if num_steps <= control_points.shape[0]:
        return control_points
    if isinstance(control_points, list):
        control_points = np.array(control_points)
    if control_points.shape[1] == 6:
        control_points_pos = control_points[:, :3]  # [N, 3]
        control_points_euler = control_points[:, 3:]  # [N, 3]
        control_points_rotmat = []
        for control_point_euler in control_points_euler:
            control_points_rotmat.append(euler2mat(control_point_euler))
        control_points_rotmat = np.array(control_points_rotmat)  # [N, 3, 3]
    elif control_points.shape[1] == 4 and control_points.shape[2] == 4:
        control_points_pos = control_points[:, :3, 3]  # [N, 3]
        control_points_rotmat = control_points[:, :3, :3]  # [N, 3, 3]
    elif control_points.shape[1] == 7:
        control_points_pos = control_points[:, :3]
        control_points_rotmat = []
        for control_point_quat in control_points[:, 3:]:
            control_points_rotmat.append(quat2mat(control_point_quat))
        control_points_rotmat = np.array(control_points_rotmat)
    else:
        raise ValueError('control_points not recognized')
    # remove the duplicate points (threshold 1e-3)
    diff = np.linalg.norm(np.diff(control_points_pos, axis=0), axis=1)
    mask = diff > 1e-3
    # always keep the first and last points
    mask = np.concatenate([[True], mask[:-1], [True]])
    control_points_pos = control_points_pos[mask]
    control_points_rotmat = control_points_rotmat[mask]
    # fit b-spline through positional terms control points
    pos_spline = fit_b_spline(control_points_pos)
    # fit RotationSpline through rotational terms control points
    times = pos_spline[1]
    rotations = R.from_matrix(control_points_rotmat)
    rot_spline = RotationSpline(times, rotations)
    # sample from the splines
    pos_samples = sample_from_spline(pos_spline, num_steps)  # [num_steps, 3]
    rot_samples = sample_from_spline(rot_spline, num_steps)  # [num_steps, 3, 3]
    if control_points.shape[1] == 6:
        poses = []
        for i in range(num_steps):
            pose = np.concatenate([pos_samples[i], mat2euler(rot_samples[i])])
            poses.append(pose)
        poses = np.array(poses)
    elif control_points.shape[1] == 4 and control_points.shape[2] == 4:
        poses = np.empty((num_steps, 4, 4))
        poses[:, :3, :3] = rot_samples
        poses[:, :3, 3] = pos_samples
        poses[:, 3, 3] = 1
    elif control_points.shape[1] == 7:
        poses = np.empty((num_steps, 7))
        for i in range(num_steps):
            quat = mat2quat(rot_samples[i])
            pose = np.concatenate([pos_samples[i], quat])
            poses[i] = pose
    return poses
    
def merge_masks(masks, ratio):
    '''
    merge masks with a max overlap ratio
    Args:
        masks: a list of np.ndarray of shape [H, W], masks
        ratio: float, max overlap ratio
    '''
    # Check whether mask i is a subset of mask j
    for i in range(len(masks)):
        if masks[i].sum() == 0:
            continue
        for j in range(i+1, len(masks)):
            if masks[j].sum() == 0:
                continue
            mask_i = masks[i]
            mask_j = masks[j]
            overlap = np.logical_and(mask_i, mask_j).sum() / mask_i.sum()
            if overlap > ratio:
                # move mask i
                masks[i] = np.zeros_like(mask_i)
                break
    # delete empty masks
    masks = [mask for mask in masks if mask.sum() > 0]
    return masks

def compute_sdf_gpu(points:np.ndarray, bounds:np.ndarray, voxel_size:float, chunk_size:int=1024)->np.ndarray:
    '''
    Compute signed distance field on GPU
    Args:
        points: np.ndarray, [N, 3], points in world frame
        bounds: np.ndarray, [2, 3], bounds in world frame
        voxel_size: float, voxel size
        chunk_size: int, chunk size for computing signed distance field
    Returns:
        np.ndarray: [H, W, D], signed distance field
    '''
    assert points.shape[-1] == 3, points.shape
    assert bounds.shape == (2, 3), bounds.shape
    assert voxel_size > 0, voxel_size
    # convert points to voxel coordinates
    min_bound = bounds[0]
    max_bound = bounds[1]

    x = torch.linspace(min_bound[0], max_bound[0], int((max_bound[0] - min_bound[0]) / voxel_size) + 1).cuda()
    y = torch.linspace(min_bound[1], max_bound[1], int((max_bound[1] - min_bound[1]) / voxel_size) + 1).cuda()
    z = torch.linspace(min_bound[2], max_bound[2], int((max_bound[2] - min_bound[2]) / voxel_size) + 1).cuda()
    shape = (len(x), len(y), len(z))

    grid = torch.stack(torch.meshgrid(x, y, z), dim=-1).reshape(-1, 3).float().cuda()
    points = torch.tensor(points).float().cuda()

    # compute signed distance field
    sdf_values = torch.empty(len(grid)).cuda()
    for i in range(0, len(grid), chunk_size):
        distances_chunk = torch.cdist(grid[i:i+chunk_size], points)
        sdf_values[i:i+chunk_size] = torch.min(distances_chunk, dim=1)[0]
    sdf_values = sdf_values.cpu().numpy().reshape(shape)

    return sdf_values

# def get_points_cloud(depth:np.ndarray, instrinsics:np.ndarray, extrinsics:np.ndarray, mask:np.ndarray)->np.ndarray:
#     '''
#     Get point cloud from depth image
#     Args:
#         depth: np.ndarray, [H, W], depth image
#         instrinsics: np.ndarray, [3, 3], camera instrinsics
#         extrinsics: np.ndarray, [4, 4], camera extrinsics
#         mask: np.ndarray, [H, W], mask
#     Returns:
#         points: list of np.ndarray, [N, 3], points in world frame
#     '''
#     assert depth.ndim == 2, depth.shape
#     assert instrinsics.shape == (3, 3), instrinsics.shape
#     assert extrinsics.shape == (4, 4), extrinsics.shape
#     assert mask.shape == depth.shape, mask.shape
#     all_points = get_points(depth, instrinsics, extrinsics)
#     object_points = []
#     objects = np.unique(mask)
#     objects = objects[objects != 0]
#     for obj in objects:
#         object_mask = (mask == obj)
#         object_points.append(all_points[object_mask])
    return object_points