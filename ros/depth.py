import argparse
import cv2
from cv_bridge import CvBridge
import os
import rosbag
import rospy
import sys

import numpy as np
import math

sys.path.append(os.path.join(os.getcwd(), "GroundingDINO"))
sys.path.append(os.path.join(os.getcwd(), "segment_anything"))


def get_average_depth(depth_img, x, y, window_size=10):
    half_size = window_size // 2
    x_min = max(x - half_size, 0)
    x_max = min(x + half_size + 1, depth_img.shape[1])
    y_min = max(y - half_size, 0)
    y_max = min(y + half_size + 1, depth_img.shape[0])

    region = depth_img[y_min:y_max, x_min:x_max]
    valid_region = region[np.isfinite(region) & (region > 0)]

    return np.mean(valid_region)


def coords_to_depth(depth_img, img_x, img_y):
    x = (img_x - cx) / fx
    y = (img_y - cy) / fy
    # z = get_average_depth(depth_img, int(x.round()), int(y.round()))
    z = depth_img[int(y.round()),  int(x.round())]
    # print(z)
    # z = depth_img.reshape(-1)[int(track_y.round()) * width + int(track_x.round())]
    x *= z
    y *= z
    return [x, y, z]


if __name__ == "__main__":

    standard_box_mask = cv2.imread('standard_box/outputs/standard_box_mask.png')
    standard_box_points_indices = np.where(standard_box_mask == 255)

    # camera info
    camera_info_file_path = os.path.join("standard_box", "camera_info.txt")
    with open(camera_info_file_path, "r") as file:
        camera_info_lines = file.readlines()
    camera_K_values = []
    for line in camera_info_lines:
        if line.strip().startswith("K:"):
            camera_K_values = eval(line.split("K:")[1].strip())
            break
    fx = camera_K_values[0]
    fy = camera_K_values[4]
    cx = camera_K_values[2]
    cy = camera_K_values[5]

    standard_box_points = []

    # process rosbag
    bag_file = os.path.join('standard_box', [f for f in os.listdir('standard_box') if f.endswith(".bag")][0])
    depth_image_topic = "/head_camera/depth/image_rect_raw"

    bridge = CvBridge()
    with rosbag.Bag(bag_file, 'r') as bag:
        start_time = bag.get_start_time()
        start_time_ros = rospy.Time(start_time)
        for topic, msg, t in bag.read_messages(topics=[depth_image_topic]):
            if t.to_sec() >= start_time_ros.to_sec() + 2:
                depth_image = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
                if msg.encoding == '16UC1':
                    depth_image = np.asarray(depth_image, dtype=np.float32)
                    depth_image /= 1000.0
                elif msg.encoding != '32FC1':
                    rospy.logerr('Unsupported depth encoding: %s' % msg.encoding)

                print(depth_image)
                for i in range(len(standard_box_points_indices[0])):
                    # print(coords_to_depth(depth_image, standard_box_points_indices[1][i], standard_box_points_indices[0][i]))
                    standard_box_points.append(coords_to_depth(depth_image, standard_box_points_indices[1][i], standard_box_points_indices[0][i]))
                break
    standard_box_points = np.array(standard_box_points)

    # transformation
    base_to_camera_transformation_translation = np.asarray([0.0970737, 0.0203574, 0.589465])
    base_to_camera_transformation_rotation = np.asarray([[-0.017472, -0.77711, 0.629122], [-0.999846, 0.014576, -0.009763], [-0.001583, -0.629196, -0.777245]])
    base_to_camera_transformation = np.eye(4)
    base_to_camera_transformation[:3, :3] = base_to_camera_transformation_rotation
    base_to_camera_transformation[:3, 3] = base_to_camera_transformation_translation

    standard_box_points_ones = np.ones((standard_box_points.shape[0], 1))
    standard_box_points_homogeneous = np.hstack([standard_box_points, standard_box_points_ones])
    standard_box_points_base_coords_homogeneous = (base_to_camera_transformation @ standard_box_points_homogeneous.T).T
    standard_box_points_base_coords = standard_box_points_base_coords_homogeneous[:, :3]

    # graph
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    x = standard_box_points_base_coords[:, 0]
    y = standard_box_points_base_coords[:, 1]
    z = standard_box_points_base_coords[:, 2]
    ax.scatter(x, y, z, c='b', marker='o')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
