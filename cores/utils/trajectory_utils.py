import numpy as np
from scipy.spatial.transform import Rotation

class PositionTrapezoidalTrajectory:
    def __init__(self, via_points, target_time, T_antp=0.2, Ts=0.01):
        """
        Create a trapezoidal velocity profile for a point-to-point motion.

        Args:
            via_points (np.array): List of via points, shape (N+1, n).
            target_time (list or np.array): List of target times for each via point, shape (N+1,).
            T_antp (float): Time to anticipate.
            Ts (float): Sampling time.
        """
            
        self.via_points = via_points
        self.target_time = target_time
        self.T_antp = T_antp
        self.Ts = Ts

        self.N = via_points.shape[0] - 1
        self.dim = via_points.shape[1]
        self.diff_vectors = np.diff(via_points, axis=0)
        self.distances = np.linalg.norm(self.diff_vectors, axis=1)

        self.start_times = np.zeros(self.N)
        self.finish_times = np.zeros(self.N)
        for i in range(self.N):
            self.start_times[i] = target_time[i] - i * T_antp
            self.finish_times[i] = target_time[i + 1] - i * T_antp

        self.t = np.linspace(target_time[0], target_time[-1], int((target_time[-1]-target_time[0]) / Ts) + 1)
        self.pd = np.zeros((len(self.t), self.dim)) + via_points[0, :]
        self.pd_dot = np.zeros((len(self.t), self.dim))
        self.pd_dot_dot = np.zeros((len(self.t), self.dim))

        for i in range(self.N):
            start_time = self.start_times[i]
            finish_time = self.finish_times[i]
            start_index = int(np.round((start_time-target_time[0]) / Ts))
            finish_index = int(np.round((finish_time-target_time[0]) / Ts))
            distance = self.distances[i]
            if distance < 1e-6:
                continue
            diff_vector = self.diff_vectors[i, :]
            for j in range(start_index, finish_index + 1):
                current_t = self.t[j]
                s, s_dot, s_dot_dot = self.trapez_vel_profile(current_t - start_time, finish_time - start_time, distance)
                self.pd[j, :] += s * diff_vector / distance
                self.pd_dot[j, :] += s_dot * diff_vector / distance
                self.pd_dot_dot[j, :] += s_dot_dot * diff_vector / distance
            if finish_index < len(self.t):
                self.pd[finish_index + 1:, :] += diff_vector


    def trapez_vel_profile(self, t, duration, distance):
        """
        Create a trapezoidal velocity profile for a point-to-point motion.

        Args:
            t (float): Current time.
            duration (float): Total duration of the motion.
            distance (float): Distance to travel.

        Returns:
            s: Position at time t.
            s_dot: Velocity at time t.
            s_dot_dot: Acceleration at time t.
        """

        if distance < 1e-6:
            return 0, 0, 0

        qc_dot_dot = 0.1
        while qc_dot_dot < 4 * distance / duration**2:
            qc_dot_dot = 2 * qc_dot_dot

        tc = duration / 2 - np.sqrt((duration**2 * qc_dot_dot - 4 * distance) / qc_dot_dot) / 2

        if 0 <= t <= tc:
            s = qc_dot_dot * t**2 / 2
            s_dot = qc_dot_dot * t
            s_dot_dot = qc_dot_dot
        elif tc < t <= duration - tc:
            s = qc_dot_dot * tc * (t - tc / 2)
            s_dot = qc_dot_dot * tc
            s_dot_dot = 0
        elif duration - tc < t <= duration:
            s = distance - qc_dot_dot * (duration - t)**2 / 2
            s_dot = qc_dot_dot * (duration - t)
            s_dot_dot = -qc_dot_dot
        elif t > duration:
            s = distance
            s_dot = 0
            s_dot_dot = 0
        else:
            raise ValueError('t out of range')

        return s, s_dot, s_dot_dot


    def get_accurate_traj_and_ders(self, t):
        """
        Create a trapezoidal velocity profile for a point-to-point motion.

        Args:
            t (float): Current time.

        Returns:
            pd (np.array): Position trajectory, shape (dim,).
            pd_dot (np.array): Velocity trajectory, shape (dim,).
            pd_dot_dot (np.array): Acceleration trajectory, shape (dim,).
        """

        pd = np.zeros(self.dim) + self.via_points[0, :]
        pd_dot = np.zeros(self.dim)
        pd_dot_dot = np.zeros(self.dim)

        for i in range(self.N):
            diff_vector = self.diff_vectors[i]
            distance = self.distances[i]
            start_time = self.start_times[i]
            finish_time = self.finish_times[i]
            duration = finish_time - start_time
            if t > finish_time:
                pd += diff_vector
            elif t>=start_time and t<=finish_time:
                s, s_dot, s_dot_dot = self.trapez_vel_profile(t - start_time, duration, distance)
                pd += s * diff_vector / distance
                pd_dot += s_dot * diff_vector / distance
                pd_dot_dot += s_dot_dot * diff_vector / distance

        return pd, pd_dot, pd_dot_dot
    
    def get_traj_and_ders(self, t):
        """
        Create a trapezoidal velocity profile for a point-to-point motion.

        Args:
            t (float): Current time.

        Returns:
            pd (np.array): Position trajectory, shape (dim,).
            pd_dot (np.array): Velocity trajectory, shape (dim,).
            pd_dot_dot (np.array): Acceleration trajectory, shape (dim,).
        """

        index = int(np.round((t-self.target_time[0]) / self.Ts))
        if index >= len(self.t):
            return self.pd[-1, :], self.pd_dot[-1, :], self.pd_dot_dot[-1, :]
        elif index >= 0 and index < len(self.t):
            return self.pd[index, :], self.pd_dot[index, :], self.pd_dot_dot[index, :]
        else:
            raise ValueError('t out of range')


class OrientationTrapezoidalTrajectory:
    def __init__(self, orientations, target_time, Ts=0.01):
        """
        Create a trapezoidal velocity profile for a point-to-point motion.

        Args:
            orientations (list of np.array): List of orientations wrt base/world frame.
            target_time (list or np.array): List of target times for each via point, shape (2,).
            Ts (float): Sampling time.
        """
            
        self.Ts = Ts
        self.target_time = target_time
        self.N = len(orientations)-1
        self.axes = np.zeros((self.N,3))
        self.angles = np.zeros(self.N)

        for i in range(self.N):
            R_i = orientations[i]
            R_f = orientations[i+1]
            R_f_i = R_i.T @ R_f
            axis_angle = np.squeeze(Rotation.from_matrix(R_f_i).as_rotvec())
            angle = np.linalg.norm(axis_angle)
            if abs(angle) > 1e-6:
                axis = axis_angle/angle
            else:
                axis = np.zeros(3)
                angle = 0

            self.axes[i,:] = axis
            self.angles[i] = angle

        self.start_times = np.zeros(self.N)
        self.finish_times = np.zeros(self.N)
        for i in range(self.N):
            self.start_times[i] = target_time[i]
            self.finish_times[i] = target_time[i + 1]

        self.t = np.linspace(target_time[0], target_time[-1], int((target_time[-1]-target_time[0]) / Ts) + 1)
        self.pd = np.zeros((len(self.t), 3, 3))
        self.pd_dot = np.zeros((len(self.t), 3))
        self.pd_dot_dot = np.zeros((len(self.t), 3))

        for i in range(self.N):
            start_time = self.start_times[i]
            finish_time = self.finish_times[i]
            start_index = int(np.round((start_time-target_time[0]) / Ts))
            finish_index = int(np.round((finish_time-target_time[0]) / Ts))
            angle = self.angles[i]
            axis = self.axes[i,:]
            rot_mat_base = orientations[i]
            for j in range(start_index, finish_index + 1):
                current_t = self.t[j]
                s, s_dot, s_dot_dot = self.trapez_vel_profile(current_t - start_time, finish_time - start_time, angle)
                self.pd[j, :, :] = rot_mat_base @ Rotation.from_rotvec(s * axis).as_matrix()
                self.pd_dot[j, :] = rot_mat_base @ (s_dot * axis)
                self.pd_dot_dot[j, :] = rot_mat_base @ (s_dot_dot * axis)
            if finish_index < len(self.t):
                self.pd[finish_index + 1:, :,:] = orientations[i+1]


    def trapez_vel_profile(self, t, duration, distance):
        """
        Create a trapezoidal velocity profile for a point-to-point motion.

        Args:
            t (float): Current time.
            duration (float): Total duration of the motion.
            distance (float): Distance to travel.

        Returns:
            s: Position at time t.
            s_dot: Velocity at time t.
            s_dot_dot: Acceleration at time t.
        """
        if abs(distance) < 1e-6:
            return 0, 0, 0

        qc_dot_dot = 0.1
        while qc_dot_dot < 4 * distance / duration**2:
            qc_dot_dot = 2 * qc_dot_dot

        tc = duration / 2 - np.sqrt((duration**2 * qc_dot_dot - 4 * distance) / qc_dot_dot) / 2

        if 0 <= t <= tc:
            s = qc_dot_dot * t**2 / 2
            s_dot = qc_dot_dot * t
            s_dot_dot = qc_dot_dot
        elif tc < t <= duration - tc:
            s = qc_dot_dot * tc * (t - tc / 2)
            s_dot = qc_dot_dot * tc
            s_dot_dot = 0
        elif duration - tc < t <= duration:
            s = distance - qc_dot_dot * (duration - t)**2 / 2
            s_dot = qc_dot_dot * (duration - t)
            s_dot_dot = -qc_dot_dot
        elif t > duration:
            s = distance
            s_dot = 0
            s_dot_dot = 0
        else:
            raise ValueError('t out of range')

        return s, s_dot, s_dot_dot

    
    def get_traj_and_ders(self, t):
        """
        Create a trapezoidal velocity profile for a point-to-point motion.

        Args:
            t (float): Current time.

        Returns:
            pd (np.array): Position trajectory, shape (3,3).
            pd_dot (np.array): Velocity trajectory, shape (3,).
            pd_dot_dot (np.array): Acceleration trajectory, shape (3,).
        """

        index = int(np.round((t-self.target_time[0]) / self.Ts))
        if index >= len(self.t):
            return self.pd[-1, :, :], self.pd_dot[-1, :], self.pd_dot_dot[-1, :]
        elif index >= 0 and index < len(self.t):
            return self.pd[index, :, :], self.pd_dot[index, :], self.pd_dot_dot[index, :]
        else:
            raise ValueError('t out of range')

class CircularTrajectory:
    def __init__(self, center_in_world, start_point_in_world, nominal_linear_velocity, R_b_to_w, start_time, end_time, Ts=0.01):
        """
        Create a circular trajectory.

        Args:
            center_in_world (np.array): center of the circular trajectory, shape (n,).
            start_point_in_world (np.array): start point of the circular trajectory, shape (n,).
            nominal_linear_velocity (float): linear velocity on the circular trajectory.
            R_b_to_w (np.array): rotation matrix from body frame to world frame, shape (n, n).
            start_time (float): start time.
            end_time (float): end time.
            Ts (float): Sampling time.
        """

        self.center_in_world = center_in_world
        self.start_point_in_world = start_point_in_world
        self.T = end_time - start_time
        self.Ts = Ts
        self.nominal_linear_velocity = nominal_linear_velocity
        self.dim = center_in_world.shape[0]
        self.R_b_to_w = R_b_to_w

        H_b_to_w = np.eye(self.dim + 1)
        H_b_to_w[:self.dim, :self.dim] = R_b_to_w
        H_b_to_w[:self.dim, self.dim] = center_in_world
        self.H_b_to_w = H_b_to_w

        self.t = np.linspace(start_time, end_time, int(self.T / Ts) + 1)
        self.pd = np.zeros((len(self.t), self.dim))
        self.pd_dot = np.zeros((len(self.t), self.dim))
        self.pd_dot_dot = np.zeros((len(self.t), self.dim))

        self.radius = np.linalg.norm(start_point_in_world - center_in_world)

        distance = self.nominal_linear_velocity * self.T
        
        start_point_in_body = np.linalg.inv(self.H_b_to_w) @ np.concatenate([start_point_in_world, np.array([1])])
        phi_offset = np.arctan2(start_point_in_body[1], start_point_in_body[0])
        for i in range(len(self.t)):
            t = self.t[i]
            s, s_dot, s_dot_dot = self.trapez_vel_profile(t- start_time, self.T, distance)
            phi = s / self.radius + phi_offset
            pd_in_body = np.array([self.radius * np.cos(phi), self.radius * np.sin(phi), 0])
            pd_dot_in_body = np.array([-s_dot * np.sin(phi), s_dot * np.cos(phi), 0])
            pd_dot_dot_in_body = np.array([-s_dot_dot * np.sin(phi) - s_dot**2 * np.cos(phi)/self.radius,
                                           s_dot_dot * np.cos(phi) - s_dot**2 * np.sin(phi)/self.radius,
                                           0])
            self.pd[i, :] = (self.H_b_to_w @ np.concatenate([pd_in_body[:self.dim], np.array([1])]))[:self.dim]
            self.pd_dot[i, :] = self.R_b_to_w @ pd_dot_in_body[:self.dim]
            self.pd_dot_dot[i, :] = self.R_b_to_w @ pd_dot_dot_in_body[:self.dim]

    
    def trapez_vel_profile(self, t, duration, distance):
        """
        Create a trapezoidal velocity profile for a point-to-point motion.

        Args:
            t (float): Current time.
            duration (float): Total duration of the motion.
            distance (float): Distance to travel.

        Returns:
            s: Position at time t.
            s_dot: Velocity at time t.
            s_dot_dot: Acceleration at time t.
        """

        qc_dot_dot = 0.1
        while qc_dot_dot < 4 * distance / duration**2:
            qc_dot_dot = 2 * qc_dot_dot

        tc = duration / 2 - np.sqrt((duration**2 * qc_dot_dot - 4 * distance) / qc_dot_dot) / 2

        if 0 <= t <= tc:
            s = qc_dot_dot * t**2 / 2
            s_dot = qc_dot_dot * t
            s_dot_dot = qc_dot_dot
        elif tc < t <= duration - tc:
            s = qc_dot_dot * tc * (t - tc / 2)
            s_dot = qc_dot_dot * tc
            s_dot_dot = 0
        elif duration - tc < t <= duration:
            s = distance - qc_dot_dot * (duration - t)**2 / 2
            s_dot = qc_dot_dot * (duration - t)
            s_dot_dot = -qc_dot_dot
        elif t > duration:
            s = distance
            s_dot = 0
            s_dot_dot = 0
        else:
            raise ValueError('t out of range')

        return s, s_dot, s_dot_dot
    
    def get_traj_and_ders(self, t):
        """
        Create a trapezoidal velocity profile for a point-to-point motion.

        Args:
            t (float): Current time.

        Returns:
            pd_in_world (np.array): Position trajectory, shape (dim,).
            pd_dot_in_world (np.array): Velocity trajectory, shape (dim,).
            pd_dot_dot_in_world (np.array): Acceleration trajectory, shape (dim,).
        """

        index = int(np.round(t / self.Ts))
        if index >= len(self.t):
            pd = self.pd[-1, :]
            pd_dot = self.pd_dot[-1, :]
            pd_dot_dot = self.pd_dot_dot[-1, :]
        elif index >= 0 and index < len(self.t):
            pd = self.pd[index, :]
            pd_dot = self.pd_dot[index, :]
            pd_dot_dot = self.pd_dot[index, :]
        else:
            raise ValueError('t out of range')

        return pd, pd_dot, pd_dot_dot

