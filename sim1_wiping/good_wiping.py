import json
import sys
import os
import argparse
import shutil
import numpy as np
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Set the backend to 'Agg'
import time
from FR3Py.sim.mujoco_with_contact import FR3Sim
from FR3Py.robot.model_collision_avoidance import PinocchioModel
from FR3Py.robot.model_collision_avoidance import BoundingShapeCoef
from cores.utils.utils import seed_everything, save_dict
from cores.utils.proxsuite_utils import init_proxsuite_qp
import diffOptHelper as doh
from cores.utils.control_utils import get_torque_to_track_traj_const_ori, get_domega_to_track_const_ori
from cores.configuration.configuration import Configuration
from scipy.spatial.transform import Rotation
from FR3Py import ASSETS_PATH
from cores.utils.trajectory_utils import TrapezoidalTrajectory, CircularTrajectory

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_num', default=2, type=int, help='test case number')
    args = parser.parse_args()

    # Create result directory
    exp_num = args.exp_num
    results_dir = "{}/sim1_results/{:03d}".format(str(Path(__file__).parent.parent), exp_num)
    test_settings_path = "{}/test_settings/test_settings_{:03d}.json".format(str(Path(__file__).parent), exp_num)
    if not os.path.exists(results_dir):
        os.mkdir(results_dir)
    shutil.copy(test_settings_path, results_dir)

    # Load test settings
    with open(test_settings_path, "r", encoding="utf8") as f:
        test_settings = json.load(f)

    # Seed everything
    seed_everything(test_settings["seed"])

    # Load configuration
    config = Configuration()

    # Various configs
    simulator_config = test_settings["simulator_config"]
    controller_config = test_settings["controller_config"]
    CBF_config = test_settings["CBF_config"]
    trajectory_config = test_settings["trajectory_config"]
    controller_config = test_settings["controller_config"]

    # Joint limits
    joint_limits_config = test_settings["joint_limits_config"]
    joint_lb = np.array(joint_limits_config["lb"], dtype=config.np_dtype)
    joint_ub = np.array(joint_limits_config["ub"], dtype=config.np_dtype)

    # Input torque limits
    input_torque_limits_config = test_settings["input_torque_limits_config"]
    input_torque_lb = np.array(input_torque_limits_config["lb"], dtype=config.np_dtype)
    input_torque_ub = np.array(input_torque_limits_config["ub"], dtype=config.np_dtype)

    # Create and reset simulation
    cam_distance = simulator_config["cam_distance"]
    cam_azimuth = simulator_config["cam_azimuth"]
    cam_elevation = simulator_config["cam_elevation"]
    cam_lookat = simulator_config["cam_lookat"]
    base_pos = np.array(simulator_config["base_pos"])
    base_quat = np.array(simulator_config["base_quat"])
    initial_joint_angles = test_settings["initial_joint_angles"]

    # Mujoco simulation
    mj_env = FR3Sim(xml_path=os.path.join(ASSETS_PATH, "mujoco/fr3_on_table_with_bounding_boxes_wiping.xml"))
    mj_env.reset(np.array(initial_joint_angles, dtype = config.np_dtype))
    mj_env.step()
    dt = mj_env.dt

    # Pinocchio model
    pin_model = PinocchioModel(base_pos=base_pos, base_quat=base_quat)

    # Load the obstacle
    obstacle_config = test_settings["obstacle_config"]
    obs_pos_2d = np.array([-0.15, 0.45], dtype=config.np_dtype)
    obs_size_2d = np.array([0.1, 0.1], dtype=config.np_dtype)
    obs_orientation_2d = np.deg2rad(0)
    obs_R_2d = np.array([[np.cos(obs_orientation_2d), -np.sin(obs_orientation_2d)],
                        [np.sin(obs_orientation_2d), np.cos(obs_orientation_2d)]], dtype=config.np_dtype) 
    obs_coef_2d = obs_R_2d @ np.diag(1/obs_size_2d**2) @ obs_R_2d.T

    obs_pos_3d = np.array([obs_pos_2d[0], obs_pos_2d[1], 0.824], dtype=config.np_dtype)
    obs_size_3d = np.array([obs_size_2d[0], obs_size_2d[1], 0.01], dtype=config.np_dtype)
    obs_R_3d = Rotation.from_euler('z', obs_orientation_2d, degrees=True).as_matrix()

    mj_env.add_visual_ellipsoid(obs_size_3d, obs_pos_3d, obs_R_3d, np.array([1,0,0,1]), id_geom_offset=0)
    id_geom_offset = mj_env.viewer.user_scn.ngeom

    # Load the bounding shape 
    eraser_bb_size_2d = np.array([0.088, 0.035])
    eraser_D_2d = np.diag(1/eraser_bb_size_2d**2)
    eraser_bb_size_3d = np.array([0.088, 0.035, 0.01])
    
    # Initial pose to pre-wiping pose
    into_surface = 0.0
    R_base_to_world = Rotation.from_quat(base_quat).as_matrix()
    P_base_to_world = base_pos
    P_EE_pre_wiping = R_base_to_world @ np.array([0.42, 0.45, into_surface]) + P_base_to_world
    P_EE_initial = R_base_to_world @ np.array([0.30, 0.0, 0.47]) + P_base_to_world
    via_points = np.array([P_EE_initial, P_EE_pre_wiping])
    target_time = np.array([0, 5])
    Ts = 0.01
    traj_line = TrapezoidalTrajectory(via_points, target_time, T_antp=0.2, Ts=Ts)

    N = 100
    len_traj = len(traj_line.t)
    sampled = np.linspace(0, len_traj-1, N).astype(int)
    sampled = traj_line.pd[sampled]
    for i in range(N-1):
        mj_env.add_visual_capsule(sampled[i], sampled[i+1], 0.004, np.array([0,0,1,1]), id_geom_offset)
        id_geom_offset = mj_env.viewer.user_scn.ngeom 
    mj_env.viewer.sync()

    # Wiping trajectory
    duration = 100
    P_center = R_base_to_world @ np.array([0.32, 0.45, into_surface]) + P_base_to_world
    nominal_linear_vel = 0.05
    circle_start_time = target_time[-1]
    circle_end_time = circle_start_time + duration
    traj_circle = CircularTrajectory(P_center, P_EE_pre_wiping, nominal_linear_vel, R_base_to_world, 
                                     circle_start_time, circle_end_time, Ts=Ts)
    N = 100
    len_traj = len(traj_circle.t)
    sampled = np.linspace(0, len_traj-1, N).astype(int)
    sampled = traj_circle.pd[sampled]
    for i in range(N-1):
        mj_env.add_visual_capsule(sampled[i], sampled[i+1], 0.004, np.array([0,0,1,1]), id_geom_offset)
        id_geom_offset = mj_env.viewer.user_scn.ngeom 
    mj_env.viewer.sync()

    # Total trajectory
    traj = np.concatenate([traj_line.pd, traj_circle.pd])
    traj_dt = np.concatenate([traj_line.pd_dot, traj_circle.pd_dot])
    traj_dtdt = np.concatenate([traj_line.pd_dot_dot, traj_circle.pd_dot_dot])
    final_end_time = traj_circle.t[-1]

    # CBF parameters
    CBF_config = test_settings["CBF_config"]
    alpha0 = CBF_config["alpha0"]
    gamma1 = CBF_config["gamma1"]
    gamma2 = CBF_config["gamma2"]
    compensation = CBF_config["compensation"] 
    selected_BBs = CBF_config["selected_bbs"]
    n_joints = 7
    n_fingers = 2

    # Define proxuite problem
    print("==> Define proxuite problem")
    n_CBF = 1
    cbf_qp = init_proxsuite_qp(n_v=2, n_eq=0, n_in=n_CBF)

    # Create records
    print("==> Create records")
    horizon = int(final_end_time/dt)
    times = np.linspace(0, (horizon-1)*dt, horizon)
    joint_angles = np.zeros([horizon, n_joints], dtype=config.np_dtype)
    finger_positions = np.zeros([horizon, n_fingers], dtype=config.np_dtype)
    controls = np.zeros([horizon, n_joints], dtype=config.np_dtype)
    desired_controls = np.zeros([horizon, 2], dtype=config.np_dtype)
    safe_controls = np.zeros([horizon, 2], dtype=config.np_dtype)
    phi1s = np.zeros([horizon, n_CBF], dtype=config.np_dtype)
    phi2s = np.zeros([horizon, n_CBF], dtype=config.np_dtype)
    cbf_values = np.zeros([horizon, n_CBF], dtype=config.np_dtype)
    time_cvxpy = np.zeros(horizon, dtype=config.np_dtype)
    time_diff_helper = np.zeros(horizon, dtype=config.np_dtype)
    time_cbf_qp = np.zeros(horizon, dtype=config.np_dtype)
    time_control_loop = np.zeros(horizon, dtype=config.np_dtype)
    all_info = []

    # Forward simulate the system
    print("==> Forward simulate the system")
    dyn_info = mj_env.getDynamicsParams()
    joint_info = mj_env.getJointStates()
    finger_info = mj_env.getFingerStates()
    q = joint_info["q"]
    dq = joint_info["dq"]
    pin_info = pin_model.getInfo(np.concatenate([joint_info["q"], finger_info["q"]]),
                                  np.concatenate([joint_info["dq"], finger_info["dq"]]))
    u_prev = np.squeeze(dyn_info["nle"][:7])
    P_EE_prev = pin_info["P_EE"]

    mj_env.add_visual_ellipsoid(eraser_bb_size_3d, pin_info["P_EE"], pin_info["R_EE"], np.array([1,0,0,0.1]), id_geom_offset=id_geom_offset)
    eraser_bb_id_offset = mj_env.viewer.user_scn.ngeom - 1
    id_geom_offset = mj_env.viewer.user_scn.ngeom 

    theta_2d = np.arctan2(pin_info["R_EE"][1,0], pin_info["R_EE"][0,0])
    R_2d_to_3d = np.array([[np.cos(theta_2d), -np.sin(theta_2d), 0],
                            [np.sin(theta_2d), np.cos(theta_2d), 0],
                            [0, 0, 1]], dtype=config.np_dtype)
    mj_env.add_visual_ellipsoid(eraser_bb_size_3d, pin_info["P_EE"], R_2d_to_3d, np.array([0,1,0,1]), id_geom_offset=id_geom_offset)
    eraser_bb_id_offset2 = mj_env.viewer.user_scn.ngeom - 1
    id_geom_offset = mj_env.viewer.user_scn.ngeom 

    for i in range(horizon):
        all_info.append({
            'dyn_info': dyn_info,
            'joint_info': joint_info,
            'finger_info': finger_info,
            'pin_info': pin_info
        })

        time_control_loop_start = time.time()

        # Update info
        dyn_info = mj_env.getDynamicsParams()
        joint_info = mj_env.getJointStates()
        finger_info = mj_env.getFingerStates()
        q = joint_info["q"]
        dq = joint_info["dq"]
        pin_info = pin_model.getInfo(np.concatenate([joint_info["q"], finger_info["q"]]),
                                  np.concatenate([joint_info["dq"], finger_info["dq"]]))

        M = pin_info["M"][0:n_joints,0:n_joints]+0.1*np.eye(n_joints) # shape (7,7)
        Minv = np.linalg.inv(M) # shape (7,7)
        nle = np.squeeze(pin_info["nle"])[0:n_joints] # shape (7,)

        # M = dyn_info["M"][0:n_joints,0:n_joints] # shape (7,7)
        # Minv = dyn_info["Minv"][0:n_joints,0:n_joints] # shape (7,7)
        # nle = np.squeeze(dyn_info["nle"][0:n_joints]) # shape (7,)

        tau_mes = joint_info["tau_est"][0:n_joints] # shape (7,)
        tau_ext = - nle + u_prev - tau_mes # shape (7,)

        P_EE = pin_info["P_EE"]
        R_EE = pin_info["R_EE"]
        J_EE = pin_info["J_EE"][:,0:n_joints] # shape (6,7)
        dJdq_EE = pin_info["dJdq_EE"] # shape (6,)
        v_EE = J_EE @ dq

        theta_2d = np.arctan2(R_EE[1,0], R_EE[0,0])

        mj_env.add_visual_ellipsoid(eraser_bb_size_3d, P_EE, R_EE, np.array([1,0,0,0.1]), id_geom_offset=eraser_bb_id_offset)

        R_2d_to_3d = np.array([[np.cos(theta_2d), -np.sin(theta_2d), 0],
                            [np.sin(theta_2d), np.cos(theta_2d), 0],
                            [0, 0, 1]], dtype=config.np_dtype)
        mj_env.add_visual_ellipsoid(eraser_bb_size_3d, pin_info["P_EE"], R_2d_to_3d, np.array([0,1,0,1]), id_geom_offset=eraser_bb_id_offset2)

        # Visualize the trajectory
        speed = np.linalg.norm((P_EE-P_EE_prev)/dt)
        rgba=np.array((np.clip(speed/10, 0, 1),
                     np.clip(1-speed/10, 0, 1),
                     .5, 1.))
        radius=.003*(1+speed)
        mj_env.add_visual_capsule(P_EE_prev, P_EE, radius, rgba, id_geom_offset, True)

        if i*dt < circle_start_time:
            # Primary obejctive: tracking control
            Kp_task = np.diag([40,40,40,100,100,100]).astype(config.np_dtype)
            Kd_task = np.diag([30,30,30,50,50,75]).astype(config.np_dtype)
            
            R_d = np.array([[1, 0, 0],
                            [0, -1, 0],
                            [0, 0, -1]], dtype=config.np_dtype)
            index = int(np.round(dt*i/Ts))
            S, u_task = get_torque_to_track_traj_const_ori(traj[index,:], traj_dt[index,:], traj_dtdt[index,:], R_d,
                                                        Kp_task, Kd_task, Minv, J_EE, dJdq_EE, dq, P_EE, R_EE)
            
            # Secondary objective: encourage the joints to remain close to the initial configuration
            W = np.diag(1.0/(joint_ub-joint_lb))
            q_bar = 1/2*(joint_ub+joint_lb)
            eq = W @ (q - q_bar)
            deq = W @ dq
            Kp_joint = 20*np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]).astype(config.np_dtype)
            Kd_joint = np.diag([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]).astype(config.np_dtype)
            u_joint = M @ (- Kd_joint @ deq - Kp_joint @ eq)
            
            # Compute the input torque
            Spinv = S.T @ np.linalg.pinv(S @ S.T + 0.01* np.eye(S.shape[0]))
            u = nle + Spinv @ u_task + (np.eye(len(q)) - Spinv @ S) @ u_joint


        if i*dt >= circle_start_time:
            index = int(np.round(dt*i/Ts))

            # Primary objective: 2D tracking while avoiding collision
            states = np.array([P_EE[0], P_EE[1], theta_2d], dtype=config.np_dtype)
            states_dt = np.array([v_EE[0], v_EE[1], v_EE[5]], dtype=config.np_dtype)
            traj_2d = np.array([traj[index,0], traj[index,1]], dtype=config.np_dtype)
            traj_2d_dt = np.array([traj_dt[index,0], traj_dt[index,1]], dtype=config.np_dtype)
            traj_2d_dtdt = np.array([traj_dtdt[index,0], traj_dtdt[index,1]], dtype=config.np_dtype)
            Kp_2d = np.diag([40,40]).astype(config.np_dtype)
            Kd_2d = np.diag([30,30]).astype(config.np_dtype)
            u_2d_nominal = traj_2d_dtdt - Kp_2d @ (states[0:2]-traj_2d) - Kd_2d @ (states_dt[0:2]-traj_2d_dt) # shape (2,)

            time_diff_helper_tmp = 0
            if (CBF_config["active"]):
                # Matrices for the CBF-QP constraints
                C = np.zeros([n_CBF, 2], dtype=config.np_dtype)
                lb = np.zeros(n_CBF, dtype=config.np_dtype)
                ub = np.zeros(n_CBF, dtype=config.np_dtype)
                CBF_tmp = np.zeros(n_CBF, dtype=config.np_dtype)
                phi1_tmp = np.zeros(n_CBF, dtype=config.np_dtype)
                phi2_tmp = np.zeros(n_CBF, dtype=config.np_dtype)

                for kk in range(n_CBF):
                    eraser_pos_2d = states[0:2]
                    eraser_theta = states[2]
                    eraser_R_2d = np.array([[np.cos(eraser_theta), -np.sin(eraser_theta)],
                                            [np.sin(eraser_theta), np.cos(eraser_theta)]], dtype=config.np_dtype)
                    time_diff_helper_tmp -= time.time()
                    alpha, _, alpha_dx_tmp, alpha_dxdx_tmp = doh.getGradientAndHessianEllipses(eraser_pos_2d, eraser_theta, eraser_D_2d,
                                                                                            eraser_R_2d, obs_coef_2d, obs_pos_2d)
                    time_diff_helper_tmp += time.time()

                    # Order of parameters in alpha_dx_tmp and alpha_dxdx_tmp: [theta, x, y]
                    # Convert to the order of [x, y, theta]
                    alpha_dx = np.zeros(3, dtype=config.np_dtype)
                    alpha_dx[0:2] = alpha_dx_tmp[1:3]
                    alpha_dx[2] = alpha_dx_tmp[0]
    
                    alpha_dxdx = np.zeros((3, 3), dtype=config.np_dtype)
                    alpha_dxdx[0:2,0:2] = alpha_dxdx_tmp[1:3,1:3]
                    alpha_dxdx[2,2] = alpha_dxdx_tmp[0,0]
                    alpha_dxdx[0:2,2] = alpha_dxdx_tmp[0,1:3]
                    alpha_dxdx[2,0:2] = alpha_dxdx_tmp[1:3,0]

                    # CBF-QP constraints
                    dCBF =  alpha_dx @ states_dt # scalar
                    CBF = alpha - alpha0[kk]
                    phi1 = dCBF + gamma1[kk] * CBF

                    C[kk,:] = alpha_dx[0:2]
                    lb[kk] = - gamma2[kk]*phi1 - gamma1[kk]*dCBF - states_dt.T @ alpha_dxdx @ states_dt + compensation[kk]
                    ub[kk] = np.inf

                    CBF_tmp[kk] = CBF
                    phi1_tmp[kk] = phi1

                # CBF-QP constraints
                g = -u_2d_nominal
                cbf_qp.update(g=g, C=C, l=lb, u=ub)
                time_cbf_qp_start = time.time()
                cbf_qp.solve()
                time_cbf_qp_end = time.time()
                u_2d = cbf_qp.results.x
                for kk in range(n_CBF):
                    phi2_tmp[kk] = C[kk,:] @ u_2d - lb[kk]
                
            else:
                u_2d = u_2d_nominal
                time_cbf_qp_start = 0
                time_cbf_qp_end = 0
                CBF_tmp = np.zeros(n_CBF, dtype=config.np_dtype)
                phi1_tmp = np.zeros(n_CBF, dtype=config.np_dtype)
                phi2_tmp = np.zeros(n_CBF, dtype=config.np_dtype)
            
            dv = np.zeros(6, dtype=config.np_dtype)
            dv[0:2] = u_2d

            # Secondary Objective: track the trajectory in z direction and keep the orientation constant
            Kp_z = 40
            Kd_z = 30
            dv[2] = traj_dtdt[index,2] - Kp_z * (P_EE[2] - traj[index,2]) - Kd_z * (v_EE[2] - traj_dt[index,2])

            Kp_ori = np.diag([200,200,200]).astype(config.np_dtype)
            Kd_ori = np.diag([50,50,50]).astype(config.np_dtype)
            R_d = np.array([[1, 0, 0],
                            [0, -1, 0],
                            [0, 0, -1]], dtype=config.np_dtype)
            dv[3:6] = get_domega_to_track_const_ori(R_d, Kp_ori, Kd_ori, v_EE[3:6], R_EE)
            S = J_EE @ Minv
            
            # Teriary Objective: encourage the joints to remain close to the initial configuration
            W = np.diag(1.0/(joint_ub-joint_lb))
            q_bar = 1/2*(joint_ub+joint_lb)
            eq = W @ (q - q_bar)
            deq = W @ dq
            Kp_joint = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])*20
            Kd_joint = np.diag([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])*1
            u_joint = M @ (- Kd_joint @ deq - Kp_joint @ eq)

            # Other 1: apply a force on the z axis to press the end-effector against the table
            F_press = np.array([0, 0, 0, 0, 0, 0])
            u_press = J_EE.T @ F_press

            # Other 2: apply a force to compensate for the friction
            mu_friction = 0.3
            F_ext = np.linalg.pinv(J_EE.T) @ tau_ext
            z_force = F_ext[2]
            F_friction = np.zeros(6)
            if z_force < 0 and np.linalg.norm(v_EE[0:2]) > 0.01:
                friction = mu_friction * np.abs(z_force)
                direction = v_EE[0:2]/np.linalg.norm(v_EE[0:2])
                F_friction[0:2] = direction * friction
            if z_force < 0 and abs(v_EE[5]) > 0.01:
                F_friction[5] = np.sign(v_EE[5]) * mu_friction * abs(z_force) * 0.03534
            u_friction = J_EE.T @ F_friction
        
            # Compute the input torque
            S_pinv = S.T @ np.linalg.pinv(S @ S.T + 0.01* np.eye(S.shape[0]))
            S_null = np.eye(n_joints) - S_pinv @ S

            u = u_press + u_friction + nle
            u += S_pinv @ (dv - dJdq_EE) 
            u += S_null @ u_joint
        
        # Step the environment
        time_control_loop_end = time.time()
        u_prev = u
        u = u - nle
        finger_pos = np.array([0.026, 0.026])
        mj_env.setCommands(u, finger_pos)
        mj_env.step()
        time.sleep(max(0,dt-time_control_loop_end+time_control_loop_start))

        # Record
        P_EE_prev = P_EE
        joint_angles[i,:] = q
        finger_positions[i,:] = finger_info["q"]
        controls[i,:] = u
        if i*dt >= circle_start_time:
            desired_controls[i,:] = u_2d_nominal
            safe_controls[i,:] = u_2d
            cbf_values[i,:] = CBF_tmp
            phi1s[i,:] = phi1_tmp
            phi2s[i,:] = phi2_tmp
            time_diff_helper[i] = time_diff_helper_tmp
            time_cbf_qp[i] = time_cbf_qp_end - time_cbf_qp_start
        time_control_loop[i] = time_control_loop_end - time_control_loop_start

    # Close the environment
    mj_env.close()

    # Save summary
    print("==> Save results")
    summary = {"times": times,
               "joint_angles": joint_angles,
               "finger_positions": finger_positions,
               "controls": controls,
               "desired_controls": desired_controls,
               "phi1s": phi1s,
               "phi2s": phi2s,
               "cbf_values": cbf_values,
               "time_cvxpy": time_cvxpy,
               "time_diff_helper": time_diff_helper,
               "time_cbf_qp": time_cbf_qp}
    save_dict(summary, os.path.join(results_dir, 'summary.pkl'))

    # print("==> Save all_info")
    # save_dict(all_info, os.path.join(results_dir, 'all_info.pkl'))
    
    # Visualization
    print("==> Draw plots")
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams.update({"text.usetex": True,
                         "text.latex.preamble": r"\usepackage{amsmath}"})
    plt.rcParams.update({'pdf.fonttype': 42})

    fig, ax = plt.subplots(figsize=(10,8), dpi=config.dpi, frameon=True)
    plt.plot(times, joint_angles[:,0], linestyle="-", label=r"$q_1$")
    plt.plot(times, joint_angles[:,1], linestyle="-", label=r"$q_2$")
    plt.plot(times, joint_angles[:,2], linestyle="-", label=r"$q_3$")
    plt.plot(times, joint_angles[:,3], linestyle="-", label=r"$q_4$")
    plt.plot(times, joint_angles[:,4], linestyle="-", label=r"$q_5$")
    plt.plot(times, joint_angles[:,5], linestyle="-", label=r"$q_6$")
    plt.plot(times, joint_angles[:,6], linestyle="-", label=r"$q_7$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'plot_joint_angles.pdf'))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10,8), dpi=config.dpi, frameon=True)
    plt.plot(times, finger_positions[:,0], linestyle="-", label=r"$q_{f1}$")
    plt.plot(times, finger_positions[:,1], linestyle="-", label=r"$q_{f2}$")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'plot_finger_positions.pdf'))
    plt.close(fig)

    for i in range(3):
        fig, ax = plt.subplots(figsize=(10,8), dpi=config.dpi, frameon=True)
        plt.plot(times, desired_controls[:,i], color="tab:blue", linestyle=":", 
                label="u_{:d} nominal".format(i+1))
        plt.plot(times, controls[:,i], color="tab:blue", linestyle="-", label="u_{:d}".format(i+1))
        plt.axhline(y = input_torque_lb[i], color = 'black', linestyle = 'dotted', linewidth = 2)
        plt.axhline(y = input_torque_ub[i], color = 'black', linestyle = 'dotted', linewidth = 2)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'plot_controls_{:d}.pdf'.format(i+1)))
        plt.close(fig)

    for i in range(n_CBF):
        fig, ax = plt.subplots(figsize=(10,8), dpi=config.dpi, frameon=True)
        plt.plot(times, phi1s[:,i], label="phi1")
        plt.plot(times, phi2s[:,i], label="phi2")
        plt.axhline(y = 0.0, color = 'black', linestyle = 'dotted', linewidth = 2)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'plot_phi_{:d}.pdf'.format(i+1)))
        plt.close(fig)
    
    for i in range(n_CBF):
        fig, ax = plt.subplots(figsize=(10,8), dpi=config.dpi, frameon=True)
        plt.plot(times, cbf_values[:,i], label="CBF")
        plt.axhline(y = 0.0, color = 'black', linestyle = 'dotted', linewidth = 2)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, 'plot_cbf_{:d}.pdf'.format(i+1)))
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(10,8), dpi=config.dpi, frameon=True)
    plt.plot(times, time_cvxpy, label="cvxpy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'plot_time_cvxpy.pdf'))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10,8), dpi=config.dpi, frameon=True)
    plt.plot(times, time_diff_helper, label="diff helper")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'plot_time_diff_helper.pdf'))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10,8), dpi=config.dpi, frameon=True)
    plt.plot(times, time_cbf_qp, label="CBF-QP")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'plot_time_cbf_qp.pdf'))
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10,8), dpi=config.dpi, frameon=True)
    plt.plot(times, time_control_loop, label="control loop")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'plot_time_control_loop.pdf'))
    plt.close(fig)

    # Print solving time
    print("==> Control loop solving time: {:.5f} s".format(np.mean(time_control_loop)))
    print("==> CVXPY solving time: {:.5f} s".format(np.mean(time_cvxpy)))
    print("==> Diff helper solving time: {:.5f} s".format(np.mean(time_diff_helper)))
    print("==> CBF-QP solving time: {:.5f} s".format(np.mean(time_cbf_qp)))

    print("==> Done!")
