from vicon.ros2 import ros2_init, ros2_close, ROS2ExecutorManager, MarkerSubscriber
import time
import pickle
import rclpy
dataset = []
def store_sample(sample):
    dataset.append(sample)
    
if __name__ == '__main__':
    ros2_init()
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
    for i in range(1000):
        rclpy.spin_once(marker_subscriber, timeout_sec=0.01)
    marker_subscriber.destroy_node()
    rclpy.shutdown()
    with open('exp3_marker_data.pickle', 'wb') as f:
        pickle.dump({'dataset':dataset}, f)
        