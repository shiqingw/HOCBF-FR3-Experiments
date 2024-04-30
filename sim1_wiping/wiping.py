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
import diffOptHelper as DOC
from cores.utils.rotation_utils import get_quat_from_rot_matrix, get_Q_matrix_from_quat, get_dQ_matrix
from cores.utils.control_utils import get_torque_to_track_traj_const_ori
from cores.configuration.configuration import Configuration
from scipy.spatial.transform import Rotation
from FR3Py import ASSETS_PATH

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp_num', default=1, type=int, help='test case number')
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
    base_pos = simulator_config["base_pos"]
    base_quat = simulator_config["base_quat"]
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
    obs_pos_np = np.array(obstacle_config["pos"], dtype=config.np_dtype)
    obs_quat_np = np.array(obstacle_config["quat"], dtype=config.np_dtype) # (x, y, z, w)
    obs_size_np = np.array(obstacle_config["size"])
    obs_coef_np = np.diag(1/obs_size_np**2)
    obs_R_np = Rotation.from_quat(obs_quat_np).as_matrix()
    obs_coef_np = obs_R_np @ obs_coef_np @ obs_R_np.T

    # Load the bounding shape coefficients
    BB_coefs = BoundingShapeCoef()
    
    # Compute desired trajectory
    traj_center = np.array(trajectory_config["center"], dtype=config.np_dtype)
    traj_radius = trajectory_config["radius"]
    traj_angular_velocity = trajectory_config["angular_velocity"]
    horizon = test_settings["horizon_length"]
    t = np.linspace(0, horizon*dt, horizon+1)
    traj = np.repeat(traj_center.reshape(1,3), horizon+1, axis=0)
    traj[:,0] += traj_radius * np.cos(traj_angular_velocity*t)
    traj[:,1] += traj_radius * np.sin(traj_angular_velocity*t)

    traj_dt = np.zeros([horizon+1, 3], dtype=config.np_dtype)
    traj_dt[:,0] = -traj_radius * traj_angular_velocity * np.sin(traj_angular_velocity*t)
    traj_dt[:,1] = traj_radius * traj_angular_velocity * np.cos(traj_angular_velocity*t)

    traj_dtdt = np.zeros([horizon+1, 3], dtype=config.np_dtype)
    traj_dtdt[:,0] = -traj_radius * traj_angular_velocity**2 * np.cos(traj_angular_velocity*t)
    traj_dtdt[:,1] = -traj_radius * traj_angular_velocity**2 * np.sin(traj_angular_velocity*t)

    # Visualize the trajectory
    N = 100
    thetas = np.linspace(0, 2*np.pi, N)
    traj_circle = np.zeros([N, 3], dtype=config.np_dtype)
    traj_circle[:,0] = traj_center[0] + traj_radius * np.cos(thetas)
    traj_circle[:,1] = traj_center[1] + traj_radius * np.sin(thetas)
    traj_circle[:,2] = 0.83
    for i in range(N-1):
        mj_env.add_visual_capsule(traj_circle[i], traj_circle[i+1], 0.004, np.array([0,0,1,1]))
        mj_env.viewer.sync()
    id_geom_offset = mj_env.viewer.user_scn.ngeom 

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
    n_CBF = len(selected_BBs)
    cbf_qp = init_proxsuite_qp(n_v=n_joints, n_eq=0, n_in=n_joints+n_CBF)

    # Create records
    print("==> Create records")
    times = np.linspace(0, (horizon-1)*dt, horizon)
    joint_angles = np.zeros([horizon, n_joints], dtype=config.np_dtype)
    finger_positions = np.zeros([horizon, n_fingers], dtype=config.np_dtype)
    controls = np.zeros([horizon, n_joints], dtype=config.np_dtype)
    desired_controls = np.zeros([horizon, n_joints], dtype=config.np_dtype)
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

        Minv_mj = dyn_info["Minv"][0:n_joints,0:n_joints] # shape (7,7)
        M_mj = dyn_info["M"][0:n_joints,0:n_joints] # shape (7,7)
        nle_mj = np.squeeze(dyn_info["nle"])[0:n_joints] # shape (7,)
        tau_mes = joint_info["tau_est"][0:n_joints] # shape (7,)

        tau_ext = np.zeros(n_joints, dtype=config.np_dtype)
        tau_ext += - nle_mj + u_prev - tau_mes 

        P_EE = pin_info["P_EE"]
        R_EE = pin_info["R_EE"]
        J_EE = pin_info["J_EE"][:,0:n_joints] # shape (6,7)
        dJdq_EE = pin_info["dJdq_EE"] # shape (6,)
        v_EE = J_EE @ dq

        # Visualize the trajectory
        speed = np.linalg.norm((P_EE-P_EE_prev)/dt)
        rgba=np.array((np.clip(speed/10, 0, 1),
                     np.clip(1-speed/10, 0, 1),
                     .5, 1.))
        radius=.003*(1+speed)
        # radius=1000*(1+speed)
        mj_env.add_visual_capsule(P_EE_prev, P_EE, radius, rgba, id_geom_offset, True)

        # Primary obejctive: tracking control
        Kp = np.diag([20,20,20,100,100,100])
        Kd = np.diag([20,20,20,100,100,100])
        
        R_d = np.array([[1, 0, 0],
                        [0, -1, 0],
                        [0, 0, -1]], dtype=config.np_dtype)
        S, u_task = get_torque_to_track_traj_const_ori(traj[i,:], traj_dt[i,:], traj_dtdt[i,:], R_d, Kp, Kd, Minv_mj, J_EE, dJdq_EE, dq, P_EE, R_EE)

        # Secondary objective: encourage the joints to remain close to the initial configuration
        W = np.diag(1.0/(joint_ub-joint_lb))
        q_bar = np.array(test_settings["initial_joint_angles"], dtype=config.np_dtype)[0:n_joints]
        eq = W @ (q - q_bar)
        deq = W @ dq
        Kp = np.diag([0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1])
        Kd = np.diag([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
        u_joint = M_mj @ (- Kd @ deq - Kp @ eq)
        
        # Compute the input torque
        Spinv = S.T @ np.linalg.pinv(S @ S.T + 0.01* np.eye(S.shape[0]))
        u_nominal = nle_mj + Spinv @ u_task + (np.eye(len(q)) - Spinv @ S) @ u_joint

        max_friction_composation = controller_config["max_friction_compensation"]
        F_ext = np.linalg.pinv(J_EE.T) @ tau_ext
        F_ext_horizontal = np.clip(F_ext.copy(), -max_friction_composation, max_friction_composation)
        F_ext_horizontal[2] = -2
        F_ext_horizontal[3] = 0
        F_ext_horizontal[4] = 0
        F_ext_horizontal[5] = 0
        tau_ext_horizontal = J_EE.T @ F_ext_horizontal
        u_nominal += tau_ext_horizontal

        time_diff_helper_tmp = 0
        if CBF_config["active"]:
            # Matrices for the CBF-QP constraints
            C = np.zeros([n_joints+n_CBF, n_joints], dtype=config.np_dtype)
            lb = np.zeros(n_joints+n_CBF, dtype=config.np_dtype)
            ub = np.zeros(n_joints+n_CBF, dtype=config.np_dtype)
            CBF_tmp = np.zeros(n_CBF, dtype=config.np_dtype)
            phi1_tmp = np.zeros(n_CBF, dtype=config.np_dtype)
            phi2_tmp = np.zeros(n_CBF, dtype=config.np_dtype)

            for kk in range(len(selected_BBs)):
                name_BB = selected_BBs[kk]
                P_BB = pin_info["P_"+name_BB]
                R_BB = pin_info["R_"+name_BB]
                J_BB = pin_info["J_"+name_BB][:,0:n_joints]
                dJdq_BB = pin_info["dJdq_"+name_BB]
                v_BB = J_BB @ dq
                D_BB = BB_coefs.coefs[name_BB]
                quat_BB = get_quat_from_rot_matrix(R_BB)
                time_diff_helper_tmp -= time.time()
                alpha, _, alpha_dx_tmp, alpha_dxdx_tmp = DOC.getGradientAndHessianEllipsoids(P_BB, quat_BB, D_BB, 
                                R_BB, obs_coef_np, obs_pos_np)
                time_diff_helper_tmp += time.time()
                
                # Order of parameters in alpha_dx_tmp and alpha_dxdx_tmp: [qx, qy, qz, qw, x, y, z]
                # Convert to the order of [x, y, z, qx, qy, qz, qw]
                alpha_dx = np.zeros(7, dtype=config.np_dtype)
                alpha_dx[0:3] = alpha_dx_tmp[4:7]
                alpha_dx[3:7] = alpha_dx_tmp[0:4]

                alpha_dxdx = np.zeros((7, 7), dtype=config.np_dtype)
                alpha_dxdx[0:3,0:3] = alpha_dxdx_tmp[4:7,4:7]
                alpha_dxdx[3:7,3:7] = alpha_dxdx_tmp[0:4,0:4]
                alpha_dxdx[0:3,3:7] = alpha_dxdx_tmp[4:7,0:4]
                alpha_dxdx[3:7,0:3] = alpha_dxdx_tmp[0:4,4:7]

                # CBF-QP constraints
                dx = np.zeros(7, dtype=config.np_dtype)
                dx[0:3] = v_BB[0:3]
                Q = get_Q_matrix_from_quat(quat_BB) # shape (4,3)
                dquat = 0.5 * Q @ v_BB[3:6] # shape (4,)
                dx[3:7] = dquat 
                dCBF =  alpha_dx @ dx # scalar
                CBF = alpha - alpha0[kk]
                phi1 = dCBF + gamma1[kk] * CBF

                dQ = get_dQ_matrix(dquat) # shape (4,3)
                tmp_vec = np.zeros(7, dtype=config.np_dtype)
                tmp_vec[3:7] = 0.5 * dQ @ v_BB[3:6] # shape (4,)
                tmp_mat = np.zeros((7,6), dtype=config.np_dtype)
                tmp_mat[0:3,0:3] = np.eye(3, dtype=config.np_dtype)
                tmp_mat[3:7,3:6] = 0.5 * Q

                C[kk,:] = alpha_dx @ tmp_mat @ J_BB @ Minv_mj
                lb[kk] = - gamma2[kk]*phi1 - gamma1[kk]*dCBF - dx.T @ alpha_dxdx @ dx - alpha_dx @ tmp_vec \
                        - alpha_dx @ tmp_mat @ dJdq_BB + alpha_dx @ tmp_mat @ J_BB @ Minv_mj @ (nle_mj + tau_ext) + compensation[kk]
                ub[kk] = np.inf

                CBF_tmp[kk] = CBF
                phi1_tmp[kk] = phi1

            # CBF-QP constraints
            g = -u_nominal
            C[n_CBF:n_CBF+n_joints,:] = np.eye(n_joints, dtype=config.np_dtype)
            lb[n_CBF:] = input_torque_lb
            ub[n_CBF:] = input_torque_ub
            cbf_qp.update(g=g, C=C, l=lb, u=ub)
            time_cbf_qp_start = time.time()
            cbf_qp.solve()
            time_cbf_qp_end = time.time()
            u = cbf_qp.results.x
            for kk in range(n_CBF):
                phi2_tmp[kk] = C[kk,:] @ u - lb[kk]
            time_control_loop_end = time.time()

        else:
            u = u_nominal
            time_cbf_qp_start = 0
            time_cbf_qp_end = 0
            time_control_loop_end = time.time()
            CBF_tmp = np.zeros(n_CBF, dtype=config.np_dtype)
            phi1_tmp = np.zeros(n_CBF, dtype=config.np_dtype)
            phi2_tmp = np.zeros(n_CBF, dtype=config.np_dtype)

        # Step the environment
        u_prev = u
        u = u - nle_mj
        finger_pos_fixed = 0.04
        mj_env.setCommands(u)
        mj_env.step()
        time.sleep(max(0,dt-time_control_loop_end+time_control_loop_start))

        # Record
        P_EE_prev = P_EE
        joint_angles[i,:] = q
        finger_positions[i,:] = finger_info["q"]
        controls[i,:] = u
        desired_controls[i,:] = u_nominal
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

    print("==> Save all_info")
    save_dict(all_info, os.path.join(results_dir, 'all_info.pkl'))
    
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

    for i in range(n_joints):
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
