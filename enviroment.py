import os
import sys
import time

import numpy as np
import cv2

import airbot

sys.path.append(os.path.dirname(__file__))
from utils.sensor import RealSense
from utils.utils import get_points
from mask_tracker import MaskTrackerProcess
from keypoint_tracker import KeypointTrackerProcess

class RealEnviroment:
    def __init__(self, config, scene_file, verbose=False, visualize=False):
        '''
        Initializes the real-world environment with configuration and scene file.
        
        Parameters:
            config (dict): Configuration dictionary for the environment.
            scene_file (str): Path to the scene file.
            verbose (bool): If True, provides additional logging.
        '''
        self.video_cache = []
        self._config = config
        self.verbose = verbose
        self.bounds_min = np.array(self.config['bounds_min'])
        self.bounds_max = np.array(self.config['bounds_max'])
        self.interpolate_pos_step_size = self.config['interpolate_pos_step_size']
        self.interpolate_rot_step_size = self.config['interpolate_rot_step_size']
        self.step_counter = 0

        self._rs = None
        self._mask_tracker = None
        self._keypoint_tracker = None

        self._num_objects = None
        self._key2obj = []

        # initialize realsense camera
        self._rs = RealSense(config['realsense'])
        self._rs.start()
        time.sleep(2)

        # ==============================
        # = Initialize mask tracker and keypoint tracker
        # ==============================
        data = None
        while data is None:
            data = self._rs.get()
        rgb = cv2.cvtColor(data['rgb'], cv2.COLOR_BGR2RGB)
        depth = data['depth'].astype(np.float32) * 0.001
        instrinsics = config['realsense']['instrinsics']
        extrinsics = data['extrinsics']
        points = get_points(depth, instrinsics, extrinsics)
        self._mask_tracker = MaskTrackerProcess(config['mask_tracker'])
        self._mask_tracker.start()
        self._mask_tracker.send(data['color'])
        masks = self._mask_tracker.get()
        self._keypoint_tracker = KeypointTrackerProcess(config['keypoint_tracker'])
        self._keypoint_tracker.start()
        self._keypoint_tracker.send({"rgb": rgb, "points": points, "masks": masks})
        self._keypoint_tracker.get()


    # ======================================
    # = exposed functions
    # ======================================
    def get_sdf_voxels(self, resolution, exclude_robot=True, exclude_obj_in_hand=True):
        '''
        Computes the signed distance field (SDF) voxels for the environment, with options to exclude certain objects.
        
        Parameters:
            resolution (float): Resolution of the SDF grid.
            exclude_robot (bool): If True, excludes robot from the SDF computation.
            exclude_obj_in_hand (bool): If True, excludes objects held by the robot from the SDF computation.
        
        Returns:
            np.ndarray: A 3D numpy array representing the SDF voxels.
        '''
        # Trivial implementation, all set to 1e6
        shape = np.ceil((self.bounds_max - self.bounds_min) / resolution).astype(int)
        sdf_voxels = np.zeros(shape) + 1e6
        return sdf_voxels

    def get_cam_obs(self):
        '''
        This method continuously attempts to retrieve data from the camera until successful.
        It processes the RGB and depth data, converts the depth data to meters, computes 3D points,
        and retrieves a mask from the mask tracker.
            dict: A dictionary containing the following keys:
            - 'rgb' (numpy.ndarray): The RGB image data.
            - 'depth' (numpy.ndarray): The depth image data in meters.
            - 'points' (numpy.ndarray): The 3D points computed from the depth data.
            - 'mask' (numpy.ndarray): The mask data from the mask tracker.
        '''
        data = None
        while data is None:
            data = self._rs.get()
        rgb = cv2.cvtColor(data['rgb'], cv2.COLOR_BGR2RGB)
        depth = data['depth'].astype(np.float32) * 0.001 # convert to meters

        instrinsics = self._config['realsense']['instrinsics']
        extrinsics = data['extrinsics']
        points = get_points(depth, instrinsics, extrinsics)

        self._mask_tracker.send(data['color'])
        mask = self._mask_tracker.get()
        return {
            'rgb': rgb,
            'depth': depth,
            'points': points,
            'mask': mask
        }

    def register_keypoints(self, keypoints):
        '''
        Registers the keypoints in the environment.
        
        Returns:
            None
        '''
        data = None
        while data is None:
            data = self._rs.get()
        rgb = cv2.cvtColor(data['rgb'], cv2.COLOR_BGR2RGB)
        depth = data['depth'].astype(np.float32) * 0.001
        instrinsics = self._config['realsense']['instrinsics']
        extrinsics = data['extrinsics']
        points = get_points(depth, instrinsics, extrinsics)
        self._keypoint_tracker.send({"rgb": rgb, "points": points, "masks": keypoints})
        res = self._keypoint_tracker.get()
        self._key2obj = res['obj_ids']
        self._num_objects = len(self._key2obj)
        

    def get_keypoint_positions(self):
        '''
        Retrieves the current positions of registered keypoints in the world frame.
        
        Returns:
            np.ndarray: Array of keypoint positions, shape (N, 3).
        '''
        while data is None:
            data = self._rs.get()
        rgb = cv2.cvtColor(data['rgb'], cv2.COLOR_BGR2RGB)
        depth = data['depth'].astype(np.float32) * 0.001
        instrinsics = self._config['realsense']['instrinsics']
        extrinsics = data['extrinsics']
        points = get_points(depth, instrinsics, extrinsics)
        self._keypoint_tracker.send({"rgb": rgb, "points": points, "masks": None})
        res = self._keypoint_tracker.get()
        return res['keypoints']    

    def get_object_by_keypoint(self, keypoint_idx):
        '''
        Retrieves the object associated with a given keypoint index.
        
        Parameters:
            keypoint_idx (int): Index of the keypoint.
        
        Returns:
            object: The object associated with the specified keypoint index.
        '''
        return self._key2obj[keypoint_idx]

    def get_collision_points(self, noise=True):
        '''
        Retrieves collision points for the robot’s gripper and any object in hand.
        
        Parameters:
            noise (bool): If True, adds noise to the collision points.
        
        Returns:
            np.ndarray: Array of collision points, shape (N, 3).
        '''
        pass

    def reset(self):
        '''
        Resets the environment and robot to their initial states.
        
        Returns:
            None
        '''
        pass

    def is_grasping(self, candidate_obj=None):
        '''
        Checks if the robot’s gripper is currently grasping an object.
        
        Parameters:
            candidate_obj (object, optional): Specific object to check for grasping.
        
        Returns:
            bool: True if the robot is grasping the object, otherwise False.
        '''
        pass

    def get_ee_pose(self):
        '''
        Retrieves the end-effector's (EE) pose, including position and orientation.
        
        Returns:
            np.ndarray: Array of [x, y, z, qx, qy, qz, qw], representing the position and orientation.
        '''
        pass

    def get_ee_pos(self):
        '''
        Retrieves the end-effector’s current position.
        
        Returns:
            np.ndarray: Array [x, y, z] representing the end-effector’s position.
        '''
        pass

    def get_ee_quat(self):
        '''
        Retrieves the end-effector’s current orientation as a quaternion.
        
        Returns:
            np.ndarray: Array [qx, qy, qz, qw] representing the orientation.
        '''
        pass

    def get_arm_joint_postions(self):
        '''
        Retrieves the current joint positions of the robot's arm.
        
        Returns:
            np.ndarray: Array of joint positions.
        '''
        pass

    def close_gripper(self):
        '''
        Closes the robot’s gripper.
        
        Returns:
            None
        '''
        pass

    def open_gripper(self):
        '''
        Opens the robot’s gripper.
        
        Returns:
            None
        '''
        pass

    def get_last_og_gripper_action(self):
        '''
        Retrieves the last action taken by the gripper.
        
        Returns:
            float: The last action taken by the gripper (e.g., 1.0 for open, 0.0 for closed).
        '''
        pass

    def get_gripper_open_action(self):
        '''
        Returns the action code for opening the gripper.
        
        Returns:
            float: Action code for opening the gripper.
        '''
        pass

    def get_gripper_close_action(self):
        '''
        Returns the action code for closing the gripper.
        
        Returns:
            float: Action code for closing the gripper.
        '''
        pass

    def get_gripper_null_action(self):
        '''
        Returns the action code for a neutral gripper action (no change).
        
        Returns:
            float: Action code for no gripper action.
        '''
        pass

    def compute_target_delta_ee(self, target_pose):
        '''
        Computes the position and rotation difference between the end-effector’s current pose and a target pose.
        
        Parameters:
            target_pose (np.ndarray): Target pose for the end-effector, shape (7,).
        
        Returns:
            tuple: (position difference, rotation difference).
        '''
        pass

    def execute_action(self, action, precise=True):
        '''
        Executes a specified action on the robot, moving the gripper to a target pose.
        
        Parameters:
            action (np.ndarray): Array of [x, y, z, qx, qy, qz, qw, gripper_action].
            precise (bool): If True, performs precise movement with small thresholds.
        
        Returns:
            tuple: (position error, rotation error) after reaching the target pose.
        '''
        pass

    def sleep(self, seconds):
        '''
        Pauses the environment’s operation for a specified number of seconds, continuing to step the simulation.
        
        Parameters:
            seconds (float): Duration of the pause in seconds.
        
        Returns:
            None
        '''
        pass

    def save_video(self, save_path=None):
        '''
        Saves the cached video frames to an MP4 file at the specified path.
        
        Parameters:
            save_path (str, optional): Path to save the video file. If None, uses default location.
        
        Returns:
            str: The path where the video file is saved.
        '''
        pass
