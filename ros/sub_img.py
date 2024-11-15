#!/usr/bin/env python
import glob
import rospy
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge
import cv2
import os
from datetime import datetime

save_dir = "image"

if not os.path.exists(save_dir):
    os.makedirs(save_dir)

bridge = CvBridge()

def image_callback(msg):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = os.path.join(save_dir, f"{timestamp}.jpg")

    try:
        # cv_image = bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        cv_image = bridge.imgmsg_to_cv2(msg, "bgr8")

        cv2.imwrite(file_name, cv_image)
        rospy.loginfo(f"Image saved as {file_name}")

    except Exception as e:
        rospy.logerr(f"Failed to save image: {e}")

def main():
    # remove jpg file
    jpg_files = glob.glob(os.path.join('image', '*.jpg'))
    for file in jpg_files:
        if os.path.isfile(file):
            os.remove(file)
            print(f'Removed file: {file}');

    rospy.init_node('image_saver', anonymous=True)
    # rospy.Subscriber("/head_camera/rgb/image_raw/compressed", CompressedImage, image_callback)
    rospy.Subscriber("/head_camera/rgb/image_raw", Image, image_callback)

    rospy.loginfo("Image saver node started, waiting for images...")
    rospy.spin()

if __name__ == '__main__':
    main()
