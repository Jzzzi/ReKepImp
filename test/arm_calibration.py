import math
import sys
import os
import time
from threading import Thread
import select

import numpy as np

import airbot
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.utils import mat2quat, quat2mat

bot = airbot.create_agent(end_mode="gripper")
time.sleep(3)
init_trans, init_quat = bot.get_current_pose()
time.sleep(3)

poses = []
tag_length = 0.074

def record_pose():
    global poses
    while True:
        if len(poses) == 4:
            print("4 point recorded! Exiting...")
            break

        trans, quat = bot.get_current_pose()
        print("\033c")
        print("Current Pose:")
        print("Translation: ")
        print(trans)
        print("Quaternion: ")
        print(quat)
        
        if select.select([sys.stdin], [], [], 0.1)[0]:
            key = sys.stdin.read(1)
            if key == 's':
                poses.append((trans, quat))
                print("Pose saved!")
            elif key == 'q':
                print("Aborted. Exiting...")
                break
        time.sleep(0.1)

def calibrate():
    '''
    return the extrinsic of the base of the arm
    '''
    global poses, tag_length
    world_points = np.array([
            [-tag_length / 2, tag_length / 2, 0],
            [tag_length / 2, tag_length / 2, 0],
            [tag_length / 2, -tag_length / 2, 0],
            [-tag_length / 2, -tag_length / 2, 0]
        ], dtype=np.float32)
    arm_points = np.array([i[0] for i in poses]).reshape(-1, 3) # (4, 3)

    # Compute centroids
    world_centroid = np.mean(world_points, axis=0)
    base_centroid = np.mean(arm_points, axis=0)

    # Center the points
    world_centered = world_points - world_centroid
    base_centered = arm_points - base_centroid

    # Compute the covariance matrix
    H = np.dot(world_centered.T, base_centered)

    # Perform Singular Value Decomposition (SVD)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # Ensure a proper rotation matrix (det(R) = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = np.dot(Vt.T, U.T)

    # Compute the translation vector
    T = base_centroid - np.dot(R, world_centroid)

    # Construct the transformation matrix
    transformation_matrix = np.eye(4)
    transformation_matrix[:3, :3] = R
    transformation_matrix[:3, 3] = T

    return transformation_matrix

if __name__ == "__main__":

    bot.manual_mode()
    record_thread = Thread(target=record_pose)
    record_thread.start()
    record_thread.join()

    bot.online_mode()
    bot.set_target_pose(init_trans, init_quat)
    time.sleep(3)

    extrinsic = calibrate()
    print("Extrinsic: ")
    print(extrinsic)

    time.sleep(3)    
