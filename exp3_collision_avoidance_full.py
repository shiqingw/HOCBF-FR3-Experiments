import time
import numpy as np
import pickle
import multiprocessing
from datetime import datetime
import os
import scipy.sparse as sparse

from FR3Py.robot.interface import FR3Real
from FR3Py.robot.model_collision_avoidance import PinocchioModel

from vicon.ros2 import ros2_init, ros2_close, ROS2ExecutorManager, MarkerSubscriber

from cores.utils.bounding_shape_coef_mj import BoundingShapeCoef
from cores.utils.osqp_utils import init_osqp
from cores.utils.rotation_utils import get_quat_from_rot_matrix

import scalingFunctionsHelperPy as sfh
import HOCBFHelperPy as hh

if __name__ == '__main__':

    # FR3 bridge
    robot = FR3Real(robot_id='fr3')
    robot_info = robot.getJointStates()
    if robot_info is None:
        raise ValueError("Robot not connected")
    print("==> FR3 bridge connected")
    
    # Pinnochio model
    pin_robot = PinocchioModel()

    # Robot parameters
    joint_lb = np.array([-2.3093, -1.5133, -2.4937, -2.7478, -2.48, 0.8521, -2.6895])
    joint_ub = np.array([2.3093, 1.5133, 2.4937, -0.4461, 2.48, 4.2094, 2.6895])
    dq_lb = np.array([-2.0, -2.0, -2.0, -2.0, -2.6100, -2.6100, -2.6100])
    dq_ub = np.array([2.0, 2.0, 2.0, 2.0, 2.6100, 2.6100, 2.6100])
    torque_lb = np.array([-87, -87, -87, -87, -12, -12, -12])
    torque_ub = np.array([87, 87, 87, 87, 12, 12, 12])
    static_friction = np.array([0.8, 2.0, 0.5, 2.0, 1.3, 1.0, 0.5])
    n_joints = 7
    n_controls = 7
    q_bar = 0.5*(joint_ub + joint_lb)
    # delta_M = np.diag([0.2, 0.25, 0.25, 0.25, 0.25, 0.25, 0.2])
    delta_M = np.zeros([n_joints, n_joints])

    # Load the bounding shape coefficients
    BB_coefs = BoundingShapeCoef()
    # selected_BBs = ["HAND_BB", "LINK7_BB", "LINK6_BB"]
    selected_BBs = ["HAND_BB", "LINK7_BB"]
    n_selected_BBs = len(selected_BBs)

    # Define SFs
    obstacle_SFs = []
    future_ind = [1, 3]
    for i in range(len(future_ind)):
        ball_radius = 0.05
        SF_ball = sfh.Ellipsoid3d(True, np.eye(3)/ball_radius**2, np.zeros(3))
        obstacle_SFs.append(SF_ball)

    robot_SFs = []
    for (i, bb_key) in enumerate(selected_BBs):
        ellipsoid_quadratic_coef = BB_coefs.coefs[bb_key]
        SF_rob = sfh.Ellipsoid3d(True, ellipsoid_quadratic_coef, np.zeros(3))
        robot_SFs.append(SF_rob)

    # Define problems
    print("==> Define diff opt problems")
    n_threads = min(max(multiprocessing.cpu_count() -1, 1), len(selected_BBs))
    probs = hh.Problem3dCollectionMovingObstacle(n_threads)
    for j in range(len(obstacle_SFs)):
        SF_obs = obstacle_SFs[j]
        obs_frame_id = j
        for i in range(len(selected_BBs)):
            SF_rob = robot_SFs[i]
            rob_frame_id = i

            prob = hh.EllipsoidAndEllipsoid3dPrb(SF_rob, SF_obs)
            probs.addProblem(prob, rob_frame_id, obs_frame_id)

    # CBF parameters
    alpha0 = 1.03
    gamma1 = 7.0
    gamma2 = 7.0
    compensation = 0.0

    # Define proxuite problem
    print("==> Define proxuite problem")
    n_obstacle = len(obstacle_SFs)
    n_robots = len(robot_SFs)
    n_CBF = n_selected_BBs*n_obstacle
    n_in = n_CBF + n_controls + n_controls + n_controls # limits on torque, dq, q
    P_diag = [1]*n_controls + [100]*(len(future_ind) - 1)
    n_v = n_controls + len(future_ind) - 1
    cbf_qp = init_osqp(n_v=n_v, n_in=n_in, P_diag=P_diag)

    # Marker subscriber
    print("==> Launch ROS2 vicon marker subscriber")
    ros2_init()
    ros2_exec_manager = ROS2ExecutorManager()
    marker_subscriber = MarkerSubscriber(user_callback=None)
    ros2_exec_manager.add_node(marker_subscriber)
    ros2_exec_manager.start()
    t_check_rate_start = time.time()
    t_check_wait = 3
    t_check_wait_count = t_check_wait
    while time.time() - t_check_rate_start < t_check_wait:
        print("Please check subscription rate...{}".format(t_check_wait_count))
        t_check_wait_count -= 1
        time.sleep(1)

    # load the extrinsic params from the pickle file
    with open('exp3_base_T_world.pkl', 'rb') as f:
        base_T_vicon = pickle.load(f)
    base_R_vicon = base_T_vicon[:3,:3].copy()

    
    # Control parameters
    q_d = np.array([0.0, -0.1, 0.0, -1.7, 0.0, 1.67, 0.785])
    Kp_joint = sparse.csc_matrix(np.diag([30.0, 30.0, 30.0, 30.0, 30.0, 30.0, 30.0]))
    Kd_joint = sparse.csc_matrix(np.diag([15.0, 15.0, 15.0, 15.0, 15.0, 15.0, 15.0]))
    
    # Main loop
    print("==> Start control loop")
    q_list = []
    dq_list = []
    tau_list = []
    all_h_list = []
    is_flying_list = []
    ball_center_list = []
    ball_vel_list = []
    time_per_loop_list = []

    try:
        t_start = time.time()
        while time.time() - t_start < 60:
            time_loop_start = time.time()

            # Robot info
            robot_info = robot.getJointStates()
            if robot_info is None:
                continue
            q = robot_info['q'] # shape (7,)
            # print(q)
            dq = robot_info['dq'] # shape (7,)
            M = robot_info['M'] + delta_M # shape (7,7)
            G = robot_info['G'] # shape (7,)
            Coriolis = robot_info['C'] # shape (7,)

            # Pinocchio info
            q_pin = 0.025*np.ones(9)
            dq_pin = np.zeros(9)
            q_pin[0:n_joints] = q # shape (9,)
            dq_pin[0:n_joints] = dq # shape (9,)
            pin_info = pin_robot.getInfo(q_pin, dq_pin)

            # Primary obejctive: tracking control
            e_joint = q - q_d
            e_joint_dot = dq
            ddq_nominal = - Kp_joint @ e_joint - Kd_joint @ e_joint_dot
            h_dq_lb = dq[:7] - dq_lb[:7]
            h_dq_ub = dq_ub[:7] - dq[:7]
            ddq_nominal = np.clip(ddq_nominal, -5*h_dq_lb, 5*h_dq_ub)

            # Ball status
            is_flying = marker_subscriber.is_flying
            all_ball_future_pos_vicon = marker_subscriber.future_pos
            all_ball_future_vel_vicon = marker_subscriber.future_vel
            
            # Collect robot and obstacle information
            all_P_rob_np = np.zeros([n_selected_BBs, 3])
            all_quat_rob_np = np.zeros([n_selected_BBs, 4])
            all_J_rob_np = np.zeros([n_selected_BBs, 6, 7])
            all_dJdq_rob_np = np.zeros([n_selected_BBs, 6])
            all_P_obs_np = np.zeros([n_obstacle, 3])
            all_quat_obs_np = np.zeros([n_obstacle, 4])
            all_v_obs_np = np.zeros([n_obstacle, 3])
            all_omega_obs_np = np.zeros([n_obstacle, 3])
            all_v_dot_obs_np = np.zeros([n_obstacle, 3])
            all_omega_dot_obs_np = np.zeros([n_obstacle, 3])

            for (ii, bb_key) in enumerate(selected_BBs):
                all_P_rob_np[ii, :] = pin_info["P_"+bb_key]
                all_J_rob_np[ii, :, :] = pin_info["J_"+bb_key][:,:7]
                all_quat_rob_np[ii, :] = get_quat_from_rot_matrix(pin_info["R_"+bb_key])
                all_dJdq_rob_np[ii, :] = pin_info["dJdq_"+bb_key][:7]
            
            for (ii, pick_idx) in enumerate(future_ind):
                ball_pos_in_vicon = all_ball_future_pos_vicon[pick_idx,:].copy()
                ball_vel_in_vicon = all_ball_future_vel_vicon[pick_idx,:].copy()
                ball_pos_in_vicon_tmp = np.array([ball_pos_in_vicon[0], ball_pos_in_vicon[1], ball_pos_in_vicon[2], 1])
                ball_pos_in_base_tmp = base_T_vicon @ ball_pos_in_vicon_tmp

                ball_pos_in_base = ball_pos_in_base_tmp[:3].copy()
                ball_vel_in_base = base_R_vicon @ ball_vel_in_vicon
                ball_acc_in_base = np.array([0,0,-9.81])
                
                all_P_obs_np[ii, :] = ball_pos_in_base
                all_quat_obs_np[ii, :] = np.array([0., 0., 0., 1.])
                all_v_obs_np[ii, :] = ball_vel_in_base
                all_omega_obs_np[ii, :] = np.zeros(3)
                all_v_dot_obs_np[ii, :] = ball_acc_in_base
                all_omega_dot_obs_np[ii, :] = np.zeros(3)

                if ii == 0:
                    ball_center_list.append(ball_pos_in_base.copy())
                    ball_vel_list.append(ball_vel_in_base.copy())

            all_h_np, all_phi1_np, all_actuation_np, all_lb_np, all_ub_np = \
                    probs.getCBFConstraints(dq, all_P_rob_np, all_quat_rob_np, all_J_rob_np, all_dJdq_rob_np, 
                                            all_P_obs_np, all_quat_obs_np, all_v_obs_np, all_omega_obs_np, 
                                            all_v_dot_obs_np, all_omega_dot_obs_np,
                                            alpha0, gamma1, gamma2, compensation)
            # print(np.min(all_h_np))

            # CBF-QP
            C = np.zeros([n_in, n_v])
            lb = np.zeros(n_in)
            ub = np.zeros(n_in)
            C[0:n_CBF,0:n_controls] = all_actuation_np
            for ii in range(len(future_ind)-1):
                C[(ii+1)*n_robots:(ii+2)*n_robots, n_controls+ii] = -np.ones(n_robots)
            lb[0:n_CBF] = all_lb_np
            ub[0:n_CBF] = all_ub_np

            # torque limits
            C[n_CBF:n_CBF+n_controls,0:n_controls] = M
            lb[n_CBF:n_CBF+n_controls] = torque_lb[:7] - Coriolis
            ub[n_CBF:n_CBF+n_controls] = torque_ub[:7] - Coriolis

            # dq limits
            h_dq_lb = dq[:7] - dq_lb[:7]
            h_dq_ub = dq_ub[:7] - dq[:7]
            C[n_CBF+n_controls:n_CBF+2*n_controls,0:n_controls] = np.eye(7)
            lb[n_CBF+n_controls:n_CBF+2*n_controls] = - 10*h_dq_lb
            ub[n_CBF+n_controls:n_CBF+2*n_controls] = 10*h_dq_ub

            # q limits
            phi1_q_lb = q[:7] - joint_lb[:7]
            dphi1_q_lb = dq[:7]
            phi2_q_lb = dphi1_q_lb + 10*phi1_q_lb

            phi1_q_ub = joint_ub[:7] - q[:7]
            dphi1_q_ub = -dq[:7]
            phi2_q_ub = dphi1_q_ub + 10*phi1_q_ub

            C[n_CBF+2*n_controls:n_CBF+3*n_controls,0:n_controls] = np.eye(7)
            lb[n_CBF+2*n_controls:n_CBF+3*n_controls] = - 10*dphi1_q_lb - 10*phi2_q_lb
            ub[n_CBF+2*n_controls:n_CBF+3*n_controls] = 10*dphi1_q_ub + 10*phi2_q_ub

            # put together
            data = C.flatten()
            rows, cols = np.indices(C.shape)
            row_indices = rows.flatten()
            col_indices = cols.flatten()
            Ax = sparse.csc_matrix((data, (row_indices, col_indices)), shape=C.shape)
            g = np.zeros(n_v)
            g[0:n_controls] = -ddq_nominal
            cbf_qp.update(q=g, l=lb, u=ub, Ax=Ax.data)
            results = cbf_qp.solve()
            ddq_safe = results.x[0:n_controls]

            # Map to torques
            try:
                if is_flying:
                    ddq = ddq_safe
                else:
                    ddq = ddq_nominal
                tau = M @ ddq + Coriolis + static_friction * np.tanh(dq)
                tau = np.clip(tau, torque_lb, torque_ub)
            finally:
                robot.setCommands(tau)
            time_loop_end = time.time()

            # Collect data
            q_list.append(q.copy())
            dq_list.append(dq.copy())
            tau_list.append(tau.copy())
            all_h_list.append(all_h_np.copy())
            time_per_loop_list.append(time_loop_end - time_loop_start)
            is_flying_list.append(is_flying)
    finally:
        robot.setCommands(np.zeros_like(tau))

        # Save data
        now = datetime.now()
        formatted_date_time = now.strftime("%Y-%m-%d-%H-%M-%S")

        directory = f"exp3_results/no_circ_{formatted_date_time}"

        # Create the directory if it doesn't exist
        if not os.path.exists(directory):
            os.makedirs(directory)            # "ddq": ddq_list,

        # Define the file path
        file_path = os.path.join(directory, "data.pickle")

        # Data to be saved
        data = {
            "q": q_list,
            "dq": dq_list,
            "tau": tau_list,
            "all_h": all_h_list,
            "is_flying": is_flying_list,
            "ball_center": ball_center_list,
            "ball_vel": ball_vel_list,
            "time_per_loop": time_per_loop_list
        }

        # Save the data using pickle
        print("==> Data saved to {}".format(file_path))
        with open(file_path, "wb") as f:
            pickle.dump(data, f)
        
        print("==> Terminate ROS2 executor manager")
        ros2_exec_manager.terminate()

    print("==> Done!")
     