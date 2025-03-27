import rospy
import yaml
from geometry_msgs.msg import Point

yaml_file_path = "updated_paper_coords.yaml"
topic_name = "/updated_paper_coords"

def read_yaml(file_path):
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def publish_point(data, publisher):
    if 't' in data:
        paper_point = data['t']
        if len(paper_point) == 3:
            point_msg = Point()
            point_msg.x, point_msg.y, point_msg.z = paper_point
            publisher.publish(point_msg)
        else:
            rospy.logwarn("tape_point_1 does not contain exactly 3 values.")
    else:
        rospy.logwarn("'t' not found in the YAML file.")

if __name__ == "__main__":
    rospy.init_node("yaml_point_publisher")

    point_pub = rospy.Publisher(topic_name, Point, queue_size=10)

    rate = rospy.Rate(1)  # 1Hz
    data = read_yaml(yaml_file_path)

    while not rospy.is_shutdown():
        publish_point(data, point_pub)
        rate.sleep()
