from vicon.ros2 import ros2_init, ros2_close, ROS2ExecutorManager, MarkerSubscriber
import time
import pickle
import rclpy
dataset = []
def store_sample(sample):
    dataset.append(sample)
    
if __name__ == '__main__':
    ros2_init()

    ros2_exec_manager = ROS2ExecutorManager()
    marker_subscriber = MarkerSubscriber(user_callback=store_sample)

    ros2_exec_manager = ROS2ExecutorManager()
    ros2_exec_manager.add_node(marker_subscriber)
    ros2_exec_manager.start()

    try:
        # Let the nodes run for 10 seconds
        time.sleep(60)
    finally:
        with open('exp3_marker_data.pickle', 'wb') as f:
            pickle.dump({'dataset':dataset}, f)
            print("data saved")
        # Terminate the executor and shutdown nodes
        ros2_exec_manager.terminate()

        