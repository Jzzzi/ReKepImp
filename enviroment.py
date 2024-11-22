import os
import sys
import time
from typing import Union, Dict
import multiprocessing as mp
import threading

import numpy as np
import cv2

import airbot
sys.path.append(os.path.dirname(__file__))
from utils.sensor import RealSense
from utils.utils import get_points, quat2mat, mat2quat, compute_sdf_gpu
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
        self._global_config = config # global config
        self._bounds_min = np.array(self._global_config['enviroment']['bounds_min'])
        self._bounds_max = np.array(self._global_config['enviroment']['bounds_max'])
        self._step_counter = 0

        # transformation matrices
        self._w2a = np.array(config['enviroment']['w2a'])
        self._a2w = np.linalg.inv(self._w2a)

        # agents
        self._arm = None
        self._rs = None
        self._mask_tracker = None
        self._keypoint_tracker = None
        self._update_thread = None
        self._stop_event = None

        self._num_objects = None
        self._objects_grasped = None
        self._key2obj = []
        self._instrinsics = None
        self._extrinsics = None

    def _update(self, stop_event):
        while not stop_event.is_set():
            self.observe(update_only=True)
            time.sleep(0.1)

    def start(self):
        # Initialize the arm
        # arm_setted = os.system("airbot_auto_set_zero")
        # assert arm_setted == 0, "Failed to set zero for the arm"
        self._arm = airbot.create_agent(end_mode="gripper")

        # Initialize RealSense
        mp.set_start_method('spawn')
        self._rs = RealSense(self._global_config['realsense'])
        self._rs.start()
        time.sleep(2) # warm up

        # Initialize mask tracker and keypoint tracker
        data = None
        while data is None:
            data = self._rs.get()
        rgb = cv2.cvtColor(data['color'], cv2.COLOR_BGR2RGB)
        depth = data['depth']
        instrinsics = data['instrinsics']
        extrinsics = data['extrinsics']
        points = get_points(depth, instrinsics, extrinsics)
        self._mask_tracker = MaskTrackerProcess(self._global_config['mask_tracker'], manual=True)
        self._mask_tracker.start()
        self._mask_tracker.send({
            'rgb': rgb,
            'points': points,
        })
        masks = self._mask_tracker.get()
        self._keypoint_tracker = KeypointTrackerProcess(self._global_config['keypoint_tracker'])
        self._keypoint_tracker.start()
        self._keypoint_tracker.send({"rgb": rgb, "points": points, "masks": masks})
        self._keypoint_tracker.get()

        # save extrinsics, instrinsics
        self._extrinsics = extrinsics
        self._instrinsics = instrinsics
        # register keypoints
        self.register_keypoints()

        self._stop_event = threading.Event()
        self._update_thread = threading.Thread(target=self._update, args=(self._stop_event,))
        self._update_thread.start()
        print(GREEN + "[RealEnviroment]: RealEnviroment started" + RESET)
    
    def observe(self, update_only = False)->Union[Dict, None]:
        '''
        Return observation while update the mask and keypoint tracker. If update_only is True, return None.
        Return:
            obs (dict): observation dict
            {
                'rgb': np.ndarray, [H, W, 3], RGB image
                'points': np.ndarray, [H, W, 3], points in world frame
                'keypoints': np.ndarray, [N, 3], keypoints in world frame
                'mask': np.ndarray, [H, W], mask image
                'projected': np.ndarray, [H, W, 3], projected image with keypoints
                'key2obj': np.ndarray, [N], keypoint to object relation
            }
        '''
        data = None
        while data is None:
            data = self._rs.get()
        rgb = cv2.cvtColor(data['color'], cv2.COLOR_BGR2RGB)
        depth = data['depth']
        instrinsics = data['instrinsics']
        extrinsics = data['extrinsics']
        points = get_points(depth, instrinsics, extrinsics)
        self._mask_tracker.send({
            "rgb": rgb,
            "points": points,
        })
        mask = self._mask_tracker.get()
        self._keypoint_tracker.send({"rgb": rgb, "points": points, "masks": mask})
        res = self._keypoint_tracker.get()
        if update_only:
            return None
        keypoints = res['keypoints']
        projected = res['projected']
        obj_ids = res['obj_ids']

        # add ee keypoint
        ee_pos = self.get_ee_pos() # ee pos in world frame xyz
        ee_keypoint = np.array([ee_pos[0], ee_pos[1], ee_pos[2]]).reshape(1, 3)
        keypoints = np.concatenate([ee_keypoint, keypoints], axis=0)
        obj_ids = np.concatenate([[1], obj_ids], axis=0)
        # get pixel coordinates of ee keypoint
        c2w = extrinsics
        w2c = np.linalg.inv(c2w)
        ee_pos_cam = np.dot(w2c[:3, :3], ee_pos) + w2c[:3, 3]
        x, y, z = ee_pos_cam
        instrinsics = instrinsics.reshape(-1)
        fx, fy, cx, cy = instrinsics[0], instrinsics[4], instrinsics[2], instrinsics[5]
        u = int(fx * x / z + cx)
        v = int(fy * y / z + cy)
        ee_pixels = (v, u)

        def _draw_zero_on_image(image, pixel):                
            if pixel is not None:
                # make sure pixel is within the image
                if pixel[0] < 0 or pixel[0] >= image.shape[0] or pixel[1] < 0 or pixel[1] >= image.shape[1]:
                    return image
                displayed_text = "0"
                text_length = len(displayed_text)
                box_width = 30 + 10 * (text_length - 1)
                box_height = 30
                cv2.rectangle(image, (pixel[1] - box_width // 2, pixel[0] - box_height // 2),
                            (pixel[1] + box_width // 2, pixel[0] + box_height // 2), (255, 255, 255), -1)
                cv2.rectangle(image, (pixel[1] - box_width // 2, pixel[0] - box_height // 2),
                            (pixel[1] + box_width // 2, pixel[0] + box_height // 2), (0, 0, 0), 2)
                org = (pixel[1] - 7 * (text_length), pixel[0] + 7)
                color = (255, 0, 0)
                image = cv2.putText(image, displayed_text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            return image

        projected = _draw_zero_on_image(projected, ee_pixels)

        return {
            'rgb': rgb,
            'points': points,
            'mask': mask,
            'keypoints': keypoints,
            'projected': projected,
            'key2obj': obj_ids
        }

    def register_keypoints(self): # keypoints para not needed
        '''
        Registers the keypoints in the environment.
        
        Returns:
            None
        '''
        observation = self.observe()
        projected = observation['projected']
        masks = observation['mask']
        keypoints = observation['keypoints']
        self._key2obj = observation['key2obj']
        print(GREEN + f"[RealEnviroment]: {len(keypoints)} keypoints registered" + RESET)
        print(GREEN + f"[RealEnviroment]: Keypoints to object relation:" + RESET)
        print(GREEN + f"{self._key2obj}" + RESET)

        # show keypoints on the image
        objs = np.unique(masks)
        objs = objs[objs != 0]
        for i in objs:
            mask_obj = (masks == i)
            color_mask = np.zeros_like(projected)
            color_mask[mask_obj] = np.array([0, 0, 255])
            projected = cv2.addWeighted(projected, 1, color_mask, 0.5, 0)
        self._num_objects = len(objs)
        self._objects_grasped = np.zeros(self._num_objects+1) # 0 meaningless, self._objects_grasped[idx] indicate whether the object idx is grasped, idx=1 is the robot
        cv2.imshow("keypoints", projected)
        cv2.waitKey(0)
        cv2.destroyWindow("keypoints")

    def get_sdf_voxels(self, resolution, exclude_robot=True, exclude_obj_in_hand=True):
        '''
        Computes the signed distance field (SDF) voxels for the environment, with options to exclude certain objects.
        
        Parameters:
            resolution (float): Resolution of the SDF grid.
            exclude_robot (bool): If True, excludes robot from the SDF computation.
            exclude_obj_in_hand (bool): If True, excludes objects held by the robot from the SDF computation.
        
        Returns:
            np.ndarray: A 3D numpy array representing the SDF voxels. Positive values are outside the object, negative values are inside.
        '''
        observation = self.observe()
        points = observation['points']
        mask  = observation['mask']
        if exclude_robot:
            # object 1 is always the robot
            robot_mask = (mask == 1)
            # expand the robot mask
            kernel = np.ones((5, 5), np.uint8)
            robot_mask = cv2.dilate(robot_mask.astype(np.uint8), kernel, iterations=1)
            points[robot_mask] = 0
        if exclude_obj_in_hand:
            hand_object_idx = np.where(self._objects_grasped == 1)[0]
            for idx in hand_object_idx:
                obj_mask = (mask == idx)
                points[obj_mask] = 0
        points = points.reshape(-1, 3)
        bounds = np.concatenate([self._bounds_min, self._bounds_max], axis=0) # [2, 3]
        # reconstruct the sdf
        sdf = compute_sdf_gpu(points, bounds, resolution)
        # reverse the sdf so that the far the object, the lower the value
        sdf = -sdf
        return sdf
    
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
        observation = self.observe()
        points = observation['points']
        mask = observation['mask']
        hand_object_idx = np.where(self._objects_grasped == 1)[0]
        collision_points = []
        for idx in hand_object_idx:
            obj_mask = (mask == idx)
            obj_points = points[obj_mask]
            # downsample to 500
            if obj_points.shape[0] > 500:
                obj_points = obj_points[np.random.choice(obj_points.shape[0], 500, replace=False)]
            collision_points.append(obj_points)
        collision_points = np.concatenate(collision_points, axis=0) # [N, 3]
        ee_points = self.get_ee_pos().reshape(1, 3)
        # sample 100 points around the ee
        if noise:
            noise = np.random.normal(0, 0.02, (100, 3))
            ee_points = np.concatenate([ee_points] * 100, axis=0) + noise
        collision_points = np.concatenate([collision_points, ee_points], axis=0)
        return collision_points

    def reset(self):
        '''
        Resets the environment and robot to their initial states.
        
        Returns:
            None
        '''
        self.start()

    def is_grasping(self, candidate_obj):
        '''
        Checks if the robot’s gripper is currently grasping an object.
        
        Parameters:
            candidate_obj (object, optional): Specific object to check for grasping.
        
        Returns:
            bool: True if the robot is grasping the object, otherwise False.
        '''
        if self._objects_grasped[candidate_obj] == 1:
            return True
        return False
    
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
        # check whether the gripper is grasping an object
        if (0.00 <= self._arm.get_current_end() and self._arm.get_current_end() <= 0.5):
            observation = self.observe()
            mask = observation['mask']
            points = observation['points']
            object_pos = []
            for obj_idx in range(1, self._num_objects+1):
                obj_mask = (mask == obj_idx)
                obj_points = points[obj_mask]
                # get the mean position of the object
                obj_pos = np.mean(obj_points, axis=0).reshape(3)
                object_pos.append(obj_pos)
            object_pos = np.array(object_pos)
            ee_pos = self.get_ee_pos()
            # set the nearest object to be grasped
            dist = np.linalg.norm(object_pos - ee_pos, axis=1)
            nearest_obj_idx = np.argmin(dist)
            self._objects_grasped = np.zeros(self._num_objects+1)
            self._objects_grasped[nearest_obj_idx + 1] = 1 # 0 meaningless, self._objects_grasped[idx] indicate whether the object idx is grasped, idx=1 is the robot
        self._last_end = 0.0

    def open_gripper(self):
        '''
        Opens the robot’s gripper.
        
        Returns:
            None
        '''
        self._arm.set_target_end(1.0)
        self._objects_grasped = np.zeros(self._num_objects+1)
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
            target_pose (np.ndarray): Target pose for the end-effector, shape (7,), translation and quaternion.
        
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
        if end >= -0.01:
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

    def stop(self):
        '''
        Terminate all processes and threads.
        '''
        # stop the update thread
        self._stop_event.set()
        self._update_thread.join()
        self._rs.stop()
        self._mask_tracker.stop()
        self._keypoint_tracker.stop()
        print(GREEN + "[RealEnviroment]: RealEnviroment stopped" + RESET)

    def get_sdf_voxels(self, resolution, exclude_robot=True, exclude_obj_in_hand=True):
        '''
        Computes the signed distance field (SDF) voxels for the environment, with options to exclude certain objects.
        
        Parameters:
            resolution (float): Resolution of the SDF grid.
            exclude_robot (bool): If True, excludes robot from the SDF computation.
            exclude_obj_in_hand (bool): If True, excludes objects held by the robot from the SDF computation.
        
        Returns:
            np.ndarray: A 3D numpy array representing the SDF voxels. Positive values are outside the object, negative values are inside.
        '''
        observation = self.observe()
        points = observation['points']
        mask  = observation['mask']
        if exclude_robot:
            # object 1 is always the robot
            robot_mask = (mask == 1)
            # expand the robot mask
            kernel = np.ones((5, 5), np.uint8)
            robot_mask = cv2.dilate(robot_mask.astype(np.uint8), kernel, iterations=1)
            points[robot_mask] = 0
        if exclude_obj_in_hand:
            hand_object_idx = np.where(self._objects_grasped == 1)[0]
            for idx in hand_object_idx:
                obj_mask = (mask == idx)
                points[obj_mask] = 0
        points = points.reshape(-1, 3)
        bounds = np.concatenate([self._bounds_min, self._bounds_max], axis=0) # [2, 3]
        # reconstruct the sdf
        sdf = compute_sdf_gpu(points, bounds, resolution)
        # reverse the sdf so that the far the object, the lower the value
        sdf = -sdf
        return sdf


    def save_video(self, save_path=None):
        '''
        Saves the cached video frames to an MP4 file at the specified path.
        
        Parameters:
            save_path (str, optional): Path to save the video file. If None, uses default location.
        
        Returns:
            str: The path where the video file is saved.
        '''
        print(YELLOW + "[RealEnviroment]: Fuck You!" + RESET)