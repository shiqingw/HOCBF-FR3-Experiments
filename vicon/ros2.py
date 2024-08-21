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

from .kalman_filter_for_ball import KalmanFilter

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
                 user_callback=None,
                 fit_with_kf=False):
        super().__init__('marker_subscriber')
        self.topic = topic
        self.subscription = self.create_subscription(
            Markers,
            self.topic,
            self.listener_callback,
            1)
        
        self.counter = 0
        
        self.user_callback = user_callback
        self.positions_np = None
        self.msg_timestamp = None

        self.center = None
        self.center_timestamp = None
        self.center_timestamp_prev = 0

        self.consecutive_frames = 0
        self.max_consecutive_frames = 48
        self.consecutive_centers = np.zeros((self.max_consecutive_frames, 3))
        self.consecutive_timestamps = np.zeros(self.max_consecutive_frames)

        self.coefficients = None
        self.c2 = None
        self.c1 = None
        self.c0 = None
        self.is_flying = False

        self.future_time = np.array([0.0, 0.05, 0.1, 0.15, 0.2, 0.25])
        self.future_pos = None
        self.future_vel = None

        self.old_stamp = time.time()

        # Initialize the Kalman filter
        dt = 1.0/240
        Q_entries = np.array([0.001, 0.001, 0.001, 10.0, 10.0, 10.0])
        Q = np.diag(Q_entries**2)
        R_entries = np.array([0.05, 0.05, 0.05])
        R = np.diag(R_entries**2)
        x0 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        P0_entries = np.array([1.0, 1.0, 1.0, 0.001, 0.001, 0.001])
        P0 = np.diag(P0_entries**2)
        self.kalman_filter = KalmanFilter(dt, Q, R, x0, P0)
        self.kf_center = None
        self.kf_velocity = None
        self.fit_with_kf = fit_with_kf

    def listener_callback(self, msg):
    
        # Extract timestamp
        self.msg_timestamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        stamp = time.time()
        self.counter += 1
        if self.counter % 100 == 0:
            self.counter = 0
            print(f"Rate: {100/(stamp-self.old_stamp)}")
            self.old_stamp = stamp

        # Extract marker positions
        positions_all = []
        for marker in msg.markers:
            if len(marker.marker_name) > 0 or len(marker.subject_name) > 0 or len(marker.segment_name) > 0:
                continue
            position = [marker.translation.x, marker.translation.y, marker.translation.z]
            positions_all.append(position)
        
        if len(positions_all) >= 4:
            positions_all_np = np.array(positions_all)/1000.0
            median = np.median(positions_all_np, axis=0)
            positions_np = positions_all_np[np.linalg.norm(positions_all_np - median, axis=1) < 0.06] # ball radius 0.04225
            self.positions_np = positions_np

        # Find the center of the sphere
        if self.positions_np is not None and len(self.positions_np) >= 4:
            self.center = self.find_sphere_center(self.positions_np)
            self.center_timestamp = self.msg_timestamp
        
        # Kalman filter updates
        self.kalman_filter.predict()
        if self.center is not None:
            self.kalman_filter.update(self.center)
            self.kf_center = self.kalman_filter.get_state()[:3]
            self.kf_velocity = self.kalman_filter.get_state()[3:6]
        
        # Update the consecutive centers
        if self.center is not None and self.center_timestamp != self.center_timestamp_prev:
            if not self.fit_with_kf:
                self.consecutive_frames = min(self.consecutive_frames + 1, self.max_consecutive_frames)
                self.consecutive_centers[:-1] = self.consecutive_centers[1:]
                self.consecutive_centers[-1,:] = self.center.copy()
                self.consecutive_timestamps[:-1] = self.consecutive_timestamps[1:]
                self.consecutive_timestamps[-1] = self.center_timestamp
                self.center_timestamp_prev = self.center_timestamp
            else:
                self.consecutive_frames = min(self.consecutive_frames + 1, self.max_consecutive_frames)
                self.consecutive_centers[:-1] = self.consecutive_centers[1:]
                self.consecutive_centers[-1,:] = self.kf_center.copy()
                self.consecutive_timestamps[:-1] = self.consecutive_timestamps[1:]
                self.consecutive_timestamps[-1] = self.center_timestamp
                self.center_timestamp_prev = self.center_timestamp
        
        # Fit the parabola to the consecutive centers
        if self.consecutive_frames == self.max_consecutive_frames:
            tmp_consecutive_timestamps = self.consecutive_timestamps - self.consecutive_timestamps[0]
            self.coefficients = self.find_coefficients(tmp_consecutive_timestamps, self.consecutive_centers)
            self.c2, self.c1, self.c0 = self.coefficients
        
        # Check if the ball is flying
        if self.c2 is not None and self.c2[-1] < -9.0 and self.c1[-1] > -10.6:
            self.is_flying = True
        else:
            self.is_flying = False

        # Predict the future position and velocity
        if self.coefficients is not None:
            future_time_tmp = self.future_time + self.consecutive_timestamps[-1] - self.consecutive_timestamps[0]
            A_pos = np.vstack([0.5 * future_time_tmp**2, future_time_tmp, np.ones_like(future_time_tmp)]).T
            self.future_pos = A_pos @ self.coefficients
            A_vel = np.vstack([future_time_tmp, np.ones_like(future_time_tmp), np.zeros_like(future_time_tmp)]).T
            self.future_vel = A_vel @ self.coefficients
        
        # Call the user-defined callback function
        if self.user_callback is not None:
            self.user_callback([self.msg_timestamp, self.center_timestamp, self.positions_np, self.center, \
                                self.c2, self.c1, self.c0, self.is_flying, self.future_pos, self.future_vel,
                                self.kf_center, self.kf_velocity])

    def find_sphere_center(self, points):
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