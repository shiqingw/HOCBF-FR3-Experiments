import threading
import time

import numpy as np
import numpy.linalg as LA

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile
from rclpy.executors import SingleThreadedExecutor, MultiThreadedExecutor
from contextlib import contextmanager

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
    def __init__(self, use_multithreading=False):
        if use_multithreading:
            self.executor = MultiThreadedExecutor()
        else:
            self.executor = SingleThreadedExecutor()
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
        self.executor_thread.daemon = True
        self.executor_thread.start()

    def terminate(self):
        """Terminate all nodes and shutdown rclpy."""
        for node in self.nodes:
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


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
        
        self.user_callback = user_callback
        self.positions_np = None
        self.timestamp = None
        self.center = None
        self.consecutive_frames = 4


        self.old_stamp = time.time()

    def listener_callback(self, msg):
    
        # Extract timestamp
        self.timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        stamp = time.time()
        print(1/(stamp - self.old_stamp))
        self.old_stamp = stamp

        # Extract marker positions
        positions_all = []
        for marker in msg.markers:
            if len(marker.marker_name) > 0 or len(marker.subject_name) > 0 or len(marker.segment_name) > 0:
                continue
            position = [marker.translation.x, marker.translation.y, marker.translation.z]
            positions_all.append(position)
        positions_all_np = np.array(positions_all)/1000.0
        median = np.median(positions_all_np, axis=0)
        positions_np = positions_all_np[np.linalg.norm(positions_all_np - median, axis=1) < 0.1]
        self.positions_np = positions_np

        if len(self.positions_np) >= 4:
            self.center = self.find_sphere_center(self.positions_np, 0.4225)

        if self.user_callback is not None:
            self.user_callback([self.timestamp, self.positions_np, self.center])

    def find_sphere_center(self, points, r):
        """
        Finds the center of a sphere given a set of non-coplanar points on its surface.

        This method calculates the center of a sphere by solving a linear system derived
        from the known points on the sphere's surface. It uses vectorized operations 
        for efficient computation.

        Parameters:
        -----------
        points : numpy.ndarray
            A 2D array of shape (N, 3), where N is the number of points and each row 
            contains the (x, y, z) coordinates of a point on the surface of the sphere.
        r : float
            The radius of the sphere. Note that the radius is not directly used in the 
            calculation but is provided to adhere to the expected function signature.

        Returns:
        --------
        numpy.ndarray
            A 1D array of shape (3,) containing the coordinates (x_c, y_c, z_c) of the 
            center of the sphere.
        """

        if len(points) < 4:
            raise ValueError("At least 4 points are required to find the sphere center.")
        
        # Choose the first point as the reference
        x1, y1, z1 = points[0]

        # Compute A matrix using vectorized operations
        A = points[1:] - points[0]

        # Compute the b vector using vectorized operations
        b = 0.5 * (np.sum(points[1:]**2, axis=1) - np.sum(points[0]**2))

        # Solve the linear system A * [x_c, y_c, z_c] = b
        center = np.linalg.lstsq(A, b, rcond=None)[0]

        return center

    def find_coefficients(self, t_samples, p_samples):
        """
        Calculate the coefficients c_2, c_1, and c_0 for the polynomial function p(t) defined as:
        
            p(t) = 1/2 * c_2 * t^2 + c_1 * t + c_0
        
        where c_2, c_1, and c_0 are vectors of size 3. The function uses a least squares method
        to determine these coefficients based on the provided samples.
        
        Parameters:
        ----------
        t_samples : numpy.ndarray
            A 1D array of time samples (shape: (N,)), where N >= 3. Each element represents
            a time point t_i at which p(t) is sampled.
        
        p_samples : numpy.ndarray
            A 2D array of p(t) samples (shape: (N, 3)), where each row corresponds to the
            3-dimensional vector p(t_i) for the corresponding time sample t_i.
        
        Returns:
        -------
        c2 : numpy.ndarray
            A 3-dimensional coefficient vector corresponding to the t^2 term in p(t) (shape: (3,)).
        
        c1 : numpy.ndarray
            A 3-dimensional coefficient vector corresponding to the t term in p(t) (shape: (3,)).
        
        c0 : numpy.ndarray
            A 3-dimensional coefficient vector corresponding to the constant term in p(t) (shape: (3,)).
        """

        if len(t_samples) < 3 or len(p_samples) < 3:
            raise ValueError("At least 3 samples are required to fit the polynomial function.")
        
        # Construct the design matrix A
        A = np.vstack([0.5 * t_samples**2, t_samples, np.ones_like(t_samples)]).T
        
        # Solve for the coefficients using the least squares method
        coefficients, _, _, _ = np.linalg.lstsq(A, p_samples, rcond=None)
        
        c2, c1, c0 = coefficients
        
        return c2, c1, c0