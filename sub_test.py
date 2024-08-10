
import time
import pickle
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from std_msgs.msg import Header
from vicon_msgs.msg import Markers, Marker
from geometry_msgs.msg import Point


class MarkerSubscriber(Node):
    def __init__(self,
                 topic='/vicon/markers',
                 user_callback=None):
        super().__init__('marker_subscriber')
        self.topic = topic
        self.subscription = self.create_subscription(
            Markers,
            self.topic,
            self.listener_callback,
            1)
        self.positions = None
        self.timestamp = None
        self.user_callback = user_callback
        self.old_stamp = time.time()

    def listener_callback(self, msg):
    
        # Extract timestamp
        self.timestamp = time.time()
        # msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        stamp = time.time()
        print(1/(stamp - self.old_stamp))
        self.old_stamp = stamp
        # Extract marker positions
        positions = []
        for marker in msg.markers:
            position = [marker.translation.x, marker.translation.y, marker.translation.z]
            positions.append(position)
        self.positions = positions
        if self.user_callback is not None:
            self.user_callback([self.timestamp, self.positions])


dataset = []
def store_sample(sample):
    dataset.append(sample)
    
if __name__ == '__main__':
    rclpy.init(args=None)
    marker_subscriber = MarkerSubscriber(user_callback=store_sample)
    # ros2_exec_manager = ROS2ExecutorManager()
    # ros2_exec_manager.add_node(marker_subscriber)
    # ros2_exec_manager.start()

    duration = 20
    N = 240*duration
    counter = 0
    timestamp_list = []
    positions_list = []
    timestamp_prev = 0
    for i in range(8000):
        rclpy.spin_once(marker_subscriber, timeout_sec=0.01)
    marker_subscriber.destroy_node()
    rclpy.shutdown()
    with open('exp3_marker_data.pickle', 'wb') as f:
        pickle.dump({'dataset':dataset}, f)