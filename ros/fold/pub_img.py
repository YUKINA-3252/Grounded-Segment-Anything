import cv2
import rospy
import os
import glob
from cv_bridge import CvBridge
from sensor_msgs.msg import Image

def get_second_newest_image(directory):
    image_files = glob.glob(os.path.join(directory, "*.jpg")) 
    image_files.sort(key=os.path.getmtime, reverse=True) 

    if len(image_files) < 2:
        rospy.logwarn("Not enough images in the directory.")
        return None

    return image_files[1]

def publish_image(directory, topic_name):
    pub = rospy.Publisher(topic_name, Image, queue_size=10)
    rospy.init_node('image_publisher', anonymous=True)
    rate = rospy.Rate(1)

    bridge = CvBridge()

    while not rospy.is_shutdown():
        image_path = get_second_newest_image(directory)
        if image_path:
            rospy.loginfo(f"Publishing: {image_path}")

            cv_image = cv2.imread(image_path)

            if cv_image is not None:
                ros_image = bridge.cv2_to_imgmsg(cv_image, encoding="bgr8")
                pub.publish(ros_image)
            else:
                rospy.logwarn(f"Failed to load image: {image_path}")
        else:
            rospy.logwarn("No image to publish.")

        rate.sleep()

if __name__ == '__main__':
    try:
        directory = "image"
        topic_name = "/head_camera/rgb/image_raw/tmp"

        publish_image(directory, topic_name)
    except rospy.ROSInterruptException:
        pass
