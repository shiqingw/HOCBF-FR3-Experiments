import numpy as np
import pickle
import matplotlib.pyplot as plt

def find_sphere_center(points, r):
    # Choose the first point as the reference
    x1, y1, z1 = points[0]

    # Compute A matrix using vectorized operations
    A = points[1:] - points[0]

    # Compute the b vector using vectorized operations
    b = 0.5 * (np.sum(points[1:]**2, axis=1) - np.sum(points[0]**2))

    # Solve the linear system A * [x_c, y_c, z_c] = b
    center = np.linalg.lstsq(A, b, rcond=None)[0]

    return center


if __name__ == '__main__':
    with open('exp3_marker_data.pickle', 'rb') as f:
        data = pickle.load(f)

    timestamps = []
    marker_positions = []
    center_online = []
    for sample in data['dataset']:
        timestamps.append(sample[0])
        marker_positions.append(sample[1])
        center_online.append(sample[2])

    timestamps = np.array(timestamps)
    timestamps -= timestamps[0]

    center = np.zeros((len(timestamps), 3))
    median = np.zeros((len(timestamps), 3))
    center_online = np.array(center_online)
    for i, positions in enumerate(marker_positions):
        positions = np.array(positions)
        median_tmp = np.median(positions, axis=0)
        median[i] = median_tmp
        positions = np.array([p for p in positions if np.linalg.norm(p - median_tmp) < 0.1])
        if len(positions) <= 4:
            center[i] = center[i-1]
        else:
            center[i] = find_sphere_center(positions, 0.4225)
    # plt.plot(timestamps, '.')
    # plt.show()

    # plt.plot(median[:, 0], label='x')
    # plt.plot(median[:, 1], label='y')
    # plt.plot(median[:, 2], label='z')
    # plt.plot(timestamps, center[:, 0], label='x')
    # plt.plot(timestamps, center[:, 1], label='y')
    # plt.plot(timestamps, center[:, 2], label='z')
    plt.plot(timestamps, center_online[:, 0], label='x')
    plt.plot(timestamps, center_online[:, 1], label='y')
    plt.plot(timestamps, center_online[:, 2], label='z')
    plt.legend()
    plt.xlabel('Time [s]')
    plt.ylabel('Position [m]')
    plt.title('Center of the sphere')
    plt.show()