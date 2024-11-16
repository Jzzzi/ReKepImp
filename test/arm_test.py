import sys
import os
import numpy as np
import time

import airbot
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.utils import mat2quat, quat2mat

if __name__ == "__main__":
    bot = airbot.create_agent(end_mode="gripper")
    data = bot.get_current_pose()
    joint = bot.get_current_joint_q()
    w2a = np.array(
        [[ 0.9994551,  -0.01450361, -0.02965028,  0.58004867],
        [ 0.01355799,  0.99940074, -0.03184862,  0.15271253],
        [ 0.03009443,  0.03142927,  0.99905281,  0.02702165],
        [ 0.        ,  0.        ,  0.        ,  1.        ]])
    rot = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
    quat = mat2quat(rot)
    if bot.add_target_relative_rotation(quat):
        while np.abs(bot.get_current_rotation() - quat).sum() > 0.01:
            time.sleep(0.5)
    # a2w = np.linalg.inv(w2a)
    while True:
        try:
            x, y, z = input("Enter the target position (x y z): ").split(' ')
        except:
            break
        x = float(x)
        y = float(y)
        z = float(z)
        target_w = np.array([x, y, z, 1])
        target_a = np.dot(w2a, target_w)
        trans_a = target_a[:3]
        rot = np.array([[1, 0, 0], [0, 0, -1], [0, 1, 0]])
        quat = mat2quat(rot)

        if not bot.set_target_pose(trans_a, quat, vel = 0.05):
            print("Failed to set target pose!")
            continue

        while True:
            if (np.abs(trans_a - bot.get_current_translation()) < 0.01).all():
                break
            time.sleep(0.5)
        print("Target reached!")
