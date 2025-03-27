#!/usr/bin/env python
import glob
import rospy
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge
import cv2
import os
from datetime import datetime

rgb_save_dir = "rgb_image"
depth_save_dir = "depth_image"

if not os.path.exists(rgb_save_dir):
    os.makedirs(rgb_save_dir)
if not os.path.exists(depth_save_dir):
    os.makedirs(depth_save_dir)

bridge = CvBridge()

def rgb_image_callback(msg):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = os.path.join(rgb_save_dir, f"{timestamp}.jpg")

    try:
        # cv_image = bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")

        cv2.imwrite(file_name, cv_image)
        rospy.loginfo(f"Image saved as {file_name}")

    except Exception as e:
        rospy.logerr(f"Failed to save image: {e}")

def depth_image_callback(msg):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = os.path.join(depth_save_dir, f"{timestamp}.png")

    try:
        depth_image = bridge.imgmsg_to_cv2(msg, desired_encoding='16UC1')
        # data = np.fromstring(msg.data, dtype=np.uint8)
        # data = data[12:]
        # depth_image = cv2.imdecode(data, cv2.IMREAD_COLOR)
        cv2.imwrite(file_name, depth_image)
        rospy.loginfo(f"Depth Image Saved as {file_name}")

    except Exception as e:
        rospy.logerr(f"Failed to save image: {e}")


def main():
    # remove jpg file
    rgb_jpg_files = glob.glob(os.path.join('rgb_image', '*.jpg'))
    depth_jpg_files = glob.glob(os.path.join('depth_image', '*.png'))
    # for file in rgb_jpg_files:
    #     if os.path.isfile(file):
    #         os.remove(file)
    #         print(f'Removed rgb file: {file}');
    # for file in depth_jpg_files:
    #     if os.path.isfile(file):
    #         os.remove(file)
    #         print(f'Removed depth file: {file}');

    rospy.init_node('image_saver', anonymous=True)
    # rospy.Subscriber("/head_camera/rgb/image_raw/compressed", CompressedImage, image_callback)
    rospy.Subscriber("/head_camera/rgb/image_raw", Image, rgb_image_callback)
    rospy.Subscriber("/head_camera/depth/image_rect_raw", Image, depth_image_callback)

    rospy.loginfo("Image saver node started, waiting for images...")
    rospy.spin()

if __name__ == '__main__':
    main()
