import time
import numpy as np

from FR3Py.robot.interface import FR3Real
from FR3Py.robot.model_collision_avoidance import PinocchioModel

from vicon.ros2 import ros2_init, ros2_close, ROS2ExecutorManager, MarkerSubscriber
import pickle

from FR3ViconVisualizer.fr3_mj_env_collision_flying_ball import FR3MuJocoEnv

if __name__ == '__main__':

    robot = FR3Real(robot_id='fr3')
    robot_info = robot.getJointStates()
    if robot_info is None:
        raise ValueError("Robot not connected")

    ros2_init()
    ros2_exec_manager = ROS2ExecutorManager()
    marker_subscriber = MarkerSubscriber(user_callback=None)
    ros2_exec_manager.add_node(marker_subscriber)
    ros2_exec_manager.start()

    env = FR3MuJocoEnv()
    env.reset()

    # load the extrinsic params from the pickle file
    with open('exp3_base_T_world.pkl', 'rb') as f:
        base_T_world = pickle.load(f)
    print(base_T_world)

    start_time = time.time()
    while time.time() -start_time < 300:
        time.sleep(0.01)
        robot_info = robot.getJointStates()
        if robot_info is None:
            continue
        q = robot_info['q']
        world_ball = marker_subscriber.kf_center
        if world_ball is None:
            continue
        world_ball = np.stack([world_ball[0], world_ball[1], world_ball[2], 1]).reshape(4, 1)
        base_ball = base_T_world @ world_ball
        T = np.eye(4)
        T[:3, 3] = base_ball[:3, 0]
        env.visualize_object(q, T)

    ros2_exec_manager.terminate()