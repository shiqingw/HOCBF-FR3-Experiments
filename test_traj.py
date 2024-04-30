import numpy as np
from cores.utils.trajectory_utils import TrapezoidalTrajectory
import matplotlib.pyplot as plt

via_points = np.array([[0, -0.8, 0],
                       [0, -0.8, 0.5],
                       [0.5, -0.6, 0.5],
                       [0.8, 0, 0.5],
                       [0.8, 0, 0]])
target_time = np.array([0, 0.6, 2.0, 3.4, 4.0])

traj = TrapezoidalTrajectory(via_points, target_time)
t = traj.t
pd = traj.pd
pd_dot = traj.pd_dot
pd_dot_dot = traj.pd_dot_dot

plt.figure()
plt.plot(t, pd)
plt.title('Position trajectory')
plt.xlabel('Time [s]')
plt.ylabel('Position [m]')
plt.grid()
plt.show()

plt.figure()
plt.plot(t, pd_dot)
plt.title('Velocity trajectory')
plt.xlabel('Time [s]')
plt.ylabel('Velocity [m/s]')
plt.grid()
plt.show()

plt.figure()
plt.plot(t, pd_dot_dot)
plt.title('Acceleration trajectory')
plt.xlabel('Time [s]')
plt.ylabel('Acceleration [m/s^2]')
plt.grid()
plt.show()
