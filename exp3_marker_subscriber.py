from vicon.ros2 import ros2_init, ros2_close, ROS2ExecutorManager, MarkerSubscriber
import time
import pickle

if __name__ == '__main__':
    ros2_init()
    marker_subscriber = MarkerSubscriber()
    ros2_exec_manager = ROS2ExecutorManager()
    ros2_exec_manager.add_node(marker_subscriber)
    ros2_exec_manager.start()

    duration = 20
    N = 240*duration
    counter = 0
    timestamp_list = []
    positions_list = []
    timestamp_prev = 0

    time_start = time.time()
    while counter < N:
        if marker_subscriber.timestamp is not None and marker_subscriber.timestamp != timestamp_prev:
            timestamp_list.append(marker_subscriber.timestamp)
            positions_list.append(marker_subscriber.positions)
            counter += 1
            timestamp_prev = marker_subscriber.timestamp
    time_end = time.time()
    print(f"Time elapsed: {time_end - time_start}")
    ros2_exec_manager.terminate()

    with open('exp3_marker_data.pickle', 'wb') as f:
        pickle.dump({'timestamps': timestamp_list, 'positions': positions_list}, f)
        