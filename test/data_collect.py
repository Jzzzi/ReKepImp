from threading import Event, Thread
from queue import Queue
import yaml
import sys
import os
import cv2
import cv2 as cv
import numpy as np
import time

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from utils.sensor import RealSense, sensor_thread

def main():
    print("Starting...")
    with open('config/config.yaml') as file:
        config = yaml.load(file, Loader=yaml.FullLoader)
    realsense_config = config['realsense']
    q = Queue()
    s = Event()
    rs = RealSense(realsense_config)
    t = Thread(target=sensor_thread, args=(q, s, rs, 3, 60))
    t.start()

    # termination criteria
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    
    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((6*8,3), np.float32)
    objp[:,:2] = np.mgrid[0:8,0:6].T.reshape(-1,2)
    
    # Arrays to store object points and image points from all the images.
    objpoints = [] # 3d point in real world space
    imgpoints = [] # 2d points in image plane.
    tic = time.time()
    while True:
        data = q.get()
        if data is None:
            break
        color = data['color']
        depth = data['depth']
        img = color
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        # Find the chess board corners
        ret, corners = cv.findChessboardCorners(gray, (8,6), None)
        if ret == True and (time.time() - tic > 0.2):
            tic = time.time()
            print("Got it!")
            corners2 = cv.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
            cv.drawChessboardCorners(img, (8,6), corners2, ret)
            objpoints.append(objp)
            imgpoints.append(corners2)

        cv.imshow('img', img)
        key = cv.waitKey(1)
                
        if key == ord('q'):
            break
    cv.destroyAllWindows()

    ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
    # show all the arguments
    # print("ret: ", ret)
    print("mtx: ", mtx)
    print("dist: ", dist)
    # print("rvecs: ", rvecs)
    # print("tvecs: ", tvecs)

    s.set()
    t.join()

if __name__ == '__main__':
    main()