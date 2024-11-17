import os
import sys
import time

import numpy as np
import cv2
import multiprocessing as mp

import airbot
sys.path.append(os.path.dirname(__file__))
from utils.sensor import RealSense
from utils.utils import get_points, quat2mat, mat2quat
from mask_tracker import MaskTrackerProcess
from keypoint_tracker import KeypointTrackerProcess

RESET = "\033[0m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"

class RealEnviroment:
    def __init__(self, config, scene_file=None, verbose=False, visualize=False):
        '''
        Initializes the real-world environment with configuration and scene file.
        
        Parameters:
            config (dict): Configuration dictionary for the environment. (Global config)
            scene_file (str): Path to the scene file.
            verbose (bool): If True, provides additional logging.
        '''
        self.video_cache = []
        self._config = config
        self.verbose = verbose
        self.bounds_min = np.array(self._config['enviroment']['bounds_min'])
        self.bounds_max = np.array(self._config['enviroment']['bounds_max'])
        # Not needed
        # self._interpolate_pos_step_size = self._config['enviroment']['interpolate_pos_step_size']
        # self._interpolate_rot_step_size = self._config['enviroment']['interpolate_rot_step_size']
        self.step_counter = 0

        self._rs = None
        self._mask_tracker = None
        self._keypoint_tracker = None

        self._num_objects = None
        self._key2obj = []

        # arm
        # arm_setted = os.system("airbot_auto_set_zero")
        # if arm_setted != 0:
        #     raise Exception("Failed to set zero for the robot arm")
        self._arm = airbot.create_agent(end_mode="gripper")
        self._w2a = np.array(config['enviroment']['w2a'])
        R = self._w2a[:3, :3]
        T = self._w2a[:3, 3]
        R_inv = R.T
        T_inv = -np.dot(R_inv, T)
        self._a2w = np.eye(4)
        self._a2w[:3, :3] = R_inv
        self._a2w[:3, 3] = T_inv

        # ==============================
        # = Initialize RealSense
        # ==============================
        mp.set_start_method('spawn')
        self._rs = RealSense(config['realsense'])
        self._rs.start()
        time.sleep(2) # warm up

        # ==============================
        # = Initialize mask tracker and keypoint tracker
        # ==============================
        data = None
        while data is None:
            data = self._rs.get()
        rgb = cv2.cvtColor(data['color'], cv2.COLOR_BGR2RGB)
        depth = data['depth']
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

        print(GREEN + "[RealEnviroment]: RealEnviroment initialized" + RESET)
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
        rgb = cv2.cvtColor(data['color'], cv2.COLOR_BGR2RGB)
        depth = data['depth']

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

    def register_keypoints(self, keypoints = None): # keypoints para not needed
        '''
        Registers the keypoints in the environment.
        
        Returns:
            None
        '''
        data = None
        while data is None:
            data = self._rs.get()
        rgb = cv2.cvtColor(data['color'], cv2.COLOR_BGR2RGB)
        depth = data['depth'].astype(np.float32)
        instrinsics = self._config['realsense']['instrinsics']
        extrinsics = data['extrinsics']
        points = get_points(depth, instrinsics, extrinsics)
        self._mask_tracker.send(data['color'])
        masks = self._mask_tracker.get()
        self._keypoint_tracker.send({"rgb": rgb, "points": points, "masks": masks})
        res = self._keypoint_tracker.get()
        self._key2obj = res['obj_ids']
        self._num_objects = len(self._key2obj)
        

    def get_keypoint_positions(self):
        '''
        Retrieves the current positions of registered keypoints in the world frame.
        
        Returns:
            np.ndarray: Array of keypoint positions, shape (N, 3).
        '''
        data = None
        while data is None:
            data = self._rs.get()
        rgb = cv2.cvtColor(data['color'], cv2.COLOR_BGR2RGB)
        depth = data['depth'].astype(np.float32)
        instrinsics = self._config['realsense']['instrinsics']
        extrinsics = data['extrinsics']
        points = get_points(depth, instrinsics, extrinsics)
        self._mask_tracker.send(data['color'])
        masks = self._mask_tracker.get()
        self._keypoint_tracker.send({"rgb": rgb, "points": points, "masks": masks})
        res = self._keypoint_tracker.get()
        return res['keypoints'], res['projected']    


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
        Retrieves collision points for the robot's gripper and any object in hand.
        
        Parameters:
            noise (bool): If True, adds noise to the collision points.
        
        Returns:
            np.ndarray: Array of collision points, shape (N, 3).
        '''
        # TODO
        pos = self.get_ee_pos()
        pos = np.dot(self._a2w, np.array([pos[0], pos[1], pos[2], 1]))[:3]
        side_length = 0.05
        points_num = 1000
        points = np.random.uniform(-side_length, side_length, (points_num, 3)) + pos
        return points

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
        return (0.01 < self._arm.get_current_end() < 0.99)
    
    def get_ee_pose(self):
        '''
        Retrieves the end-effector's (EE) pose, including position and orientation.
        
        Returns:
            np.ndarray: Array of [x, y, z, qx, qy, qz, qw], representing the position and orientation.
        '''
        pos, quat = self._arm.get_current_pose()
        # convert to world coordinate
        pos = np.dot(self._a2w, np.array([pos[0], pos[1], pos[2], 1]))[:3]
        mat = quat2mat(quat)
        mat = np.dot(self._a2w[:3, :3], mat)
        quat = mat2quat(mat)
        return np.concatenate([pos, quat])    

    def get_ee_pos(self):
        '''
        Retrieves the end-effector’s current position.
        
        Returns:
            np.ndarray: Array [x, y, z] representing the end-effector’s position.
        '''
        return self.get_ee_pose()[:3]

    def get_ee_quat(self):
        '''
        Retrieves the end-effector’s current orientation as a quaternion.
        
        Returns:
            np.ndarray: Array [qx, qy, qz, qw] representing the orientation.
        '''
        return self.get_ee_pose()[3:]


    def get_arm_joint_postions(self):
        '''
        Retrieves the current joint positions of the robot's arm.
        
        Returns:
            np.ndarray: Array of joint positions.
        '''
        return self._arm.get_current_joint_q()

    def close_gripper(self):
        '''
        Closes the robot’s gripper.
        
        Returns:
            None
        '''
        self._arm.set_target_end(0.0)
        self._last_end = 0.0

    def open_gripper(self):
        '''
        Opens the robot’s gripper.
        
        Returns:
            None
        '''
        self._arm.set_target_end(1.0)
        self._last_end = 1.0

    def get_last_og_gripper_action(self):
        '''
        Retrieves the last action taken by the gripper.
        
        Returns:
            float: The last action taken by the gripper (e.g., 1.0 for open, 0.0 for closed).
        '''
        try:
            return self._last_end
        except:
            # if no action has been taken yet
            return None

    def get_gripper_open_action(self):
        '''
        Returns the action code for opening the gripper.
        
        Returns:
            float: Action code for opening the gripper.
        '''
        return 1.0

    def get_gripper_close_action(self):
        '''
        Returns the action code for closing the gripper.
        
        Returns:
            float: Action code for closing the gripper.
        '''
        return 0.0

    def get_gripper_null_action(self):
        '''
        Returns the action code for a neutral gripper action (no change).
        
        Returns:
            float: Action code for no gripper action.
        '''
        return -1.0

    def compute_target_delta_ee(self, target_pose):
        '''
        Computes the position and rotation difference between the end-effector’s current pose and a target pose.
        
        Parameters:
            target_pose (np.ndarray): Target pose for the end-effector, shape (7,).
        
        Returns:
            tuple: (position difference, rotation difference).
        '''
        pos, quat = self.get_ee_pose()[:3], self.get_ee_pose()[3:]
        target_pos = target_pose[:3]
        target_quat = target_pose[3:]
        pos_diff = target_pos - pos
        rot_diff = target_quat - quat
        return pos_diff, rot_diff

    def execute_action(self, action, precise=True, wait=False):
        '''
        Executes a specified action on the robot, moving the gripper to a target pose.
        
        Parameters:
            action (np.ndarray): Array of [x, y, z, qx, qy, qz, qw, gripper_action].
            precise (bool): If True, performs precise movement with small thresholds.
            wait (bool): If True, waits for the robot to reach the target pose before returning.
        
        Returns:
            tuple: (position error, rotation error) after reaching the target pose.
        '''
        pos = action[:3]
        quat = action[3:7]
        end = action[7]
        # convert to arm coordinate
        pos = np.dot(self._w2a, np.array([pos[0], pos[1], pos[2], 1]))[:3]
        mat = quat2mat(quat)
        mat = np.dot(self._w2a[:3, :3], mat)
        quat = mat2quat(mat)

        reachable = self._arm.set_target_pose(pos, quat, vel = 0.05)
        if not reachable:
            return False
        self._arm.set_target_end(end)
        if wait:
            while True:
                if (np.abs(pos - self._arm.get_current_translation()) < 0.01).all():
                    if (np.abs(quat - self._arm.get_current_rotation()) < 0.01).all():
                        return True
                time.sleep(0.5)
        return True

    def sleep(self, seconds):
        '''
        Pauses the environment’s operation for a specified number of seconds, continuing to step the simulation.
        
        Parameters:
            seconds (float): Duration of the pause in seconds.
        
        Returns:
            None
        '''
        time.sleep(seconds)

    def save_video(self, save_path=None):
        '''
        Saves the cached video frames to an MP4 file at the specified path.
        
        Parameters:
            save_path (str, optional): Path to save the video file. If None, uses default location.
        
        Returns:
            str: The path where the video file is saved.
        '''
        print(YELLOW + "[RealEnviroment]: Fuck You!" + RESET)

    def stop(self):
        '''
        Terminate all processes and threads.
        '''
        self._rs.stop()
        self._mask_tracker.stop()
        self._keypoint_tracker.stop()
        print(GREEN + "[RealEnviroment]: RealEnviroment stopped" + RESET)
