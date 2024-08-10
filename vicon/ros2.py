import struct
import threading
import time

import numpy as np
import numpy.linalg as LA

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.executors import MultiThreadedExecutor
from std_msgs.msg import Header
from vicon_msgs.msg import Markers, Marker
from geometry_msgs.msg import Point


def ros2_init(args=None):
    rclpy.init(args=args)


def ros2_close():
    if rclpy.ok():
        rclpy.shutdown()


class ROS2ExecutorManager:
    """A class to manage the ROS2 executor. It allows to add nodes and start the executor in a separate thread."""
    def __init__(self):
        self.executor = MultiThreadedExecutor()
        self.nodes = []
        self.executor_thread = None

    def add_node(self, node: Node):
        """Add a new node to the executor."""
        self.nodes.append(node)
        self.executor.add_node(node)

    def _run_executor(self):
        try:
            self.executor.spin()
        except KeyboardInterrupt:
            pass
        finally:
            self.terminate()

    def start(self):
        """Start spinning the nodes in a separate thread."""
        self.executor_thread = threading.Thread(target=self._run_executor)
        self.executor_thread.start()

    def terminate(self):
        """Terminate all nodes and shutdown rclpy."""
        for node in self.nodes:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if self.executor_thread and threading.current_thread() != self.executor_thread:
            self.executor_thread.join()


class MarkerSubscriber(Node):
    def __init__(self,
                 topic='/vicon/markers'
                 ):
        super().__init__('marker_subscriber')
        self.topic = topic
        self.subscription = self.create_subscription(
            Markers,
            self.topic,
            self.listener_callback,
            1)
        self.positions = None
        self.timestamp = None

        self.last_time = self.get_clock().now()
        self.counter = 0

    def listener_callback(self, msg):
        
        self.counter += 1
        if self.counter % 100 == 0:  # Print every 100 messages
            current_time = self.get_clock().now()
            time_diff = (current_time - self.last_time).nanoseconds / 1e9  # Convert nanoseconds to seconds
            self.last_time = current_time
            frequency = 100 / time_diff
            self.get_logger().info(f'Callback frequency: {frequency:.2f} Hz')

        # Extract timestamp
        self.timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        
        # Extract marker positions
        positions = []
        for marker in msg.markers:
            position = [marker.translation.x, marker.translation.y, marker.translation.z]
            positions.append(position)
        self.positions = positions

