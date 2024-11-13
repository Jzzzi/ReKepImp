import cv2
import numpy as np

instrinsics = [386.74, 0, 326.66, 0, 386.28, 247.33, 0, 0, 1]
instrinsics = np.array(instrinsics).reshape(3, 3)
distortion = [0.0, 0.0, 0.0, 0.0, 0.0]
distortion = np.array(distortion)

dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_16H5)
params = cv2.aruco.DetectorParameters()
detector = cv2.aruco.ArucoDetector(dict, params)
length = 0.074

color = cv2.imread('color.png')
img = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)

corners, ids, _ = detector.detectMarkers(img)
corners = np.array(corners)
ids = np.array(ids)


if ids is not None and len(corners) > 0:
    
    # 定义 3D 世界坐标系的标记角点 (以米为单位)
    obj_points = np.array([
        [-length / 2, -length / 2, 0],
        [length / 2, -length / 2, 0],
        [length / 2, length / 2, 0],
        [-length / 2, length / 2, 0]
    ])

    # 使用 solvePnP 估计每个标记的姿态
    rvecs = []
    tvecs = []

    for i, corner in enumerate(corners):
        # 获取检测到的标记的四个角点
        print(corner.shape)
        img_points = corner[0].reshape(-1, 2)

        # 使用 solvePnP 计算旋转和平移向量
        success, rvec, tvec = cv2.solvePnP(obj_points, img_points, instrinsics, distortion)

        if success:
            rvecs.append(rvec)
            tvecs.append(tvec)

            # 打印姿态估计结果
            print(f"Marker ID: {ids[i][0]}")
            print(f"Rotation Vector (rvec):\n{rvec}")
            print(f"Translation Vector (tvec):\n{tvec}")

        color = cv2.aruco.drawDetectedMarkers(color, corners, ids)

        # 显示带有检测结果的图像
        cv2.imshow("Pose Estimation", color)
        cv2.waitKey(0)
        # cv2.destroyAllWindows()
else:
    print("No markers detected.")