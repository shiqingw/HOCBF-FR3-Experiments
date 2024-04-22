import numpy as np
from cores.utils.utils import get_skew_symmetric_matrix
from cores.utils.rotation_utils import get_quat_from_rot_matrix

from cores.configuration.configuration import Configuration
config = Configuration()

def solve_infinite_LQR(A, B, Q, R):
    '''
    A, B, Q and R are the matrices defining the OC problem
    QN is the matrix used for the terminal cost
    N is the horizon length
    '''
    tol = 1e-5
    P_old = np.zeros(Q.shape)
    P = np.eye(Q.shape[0])
    while np.linalg.norm(P - P_old) > tol: 
        P_old = P
        T = B.T @ P @ B + R
        K = - np.linalg.inv(T) @ B.T @ P @ A
        P = Q + A.T @ P @ A + A.T @ P @ B @ K
    return K

def solve_LQR_tracking(A_list, B_list, Q_list, R_list, x_bar, N):
    '''
    A_list, B_list, Q_list and R_list are the matrices defining the OC problem
    x_bar is the trajectory of desired states of size dim(x) x (N+1)
    N is the horizon length
    
    The function returns 1) a list of gains of length N and 2) a list of feedforward controls of length N
    '''
    Q = Q_list[-1]
    P = Q
    q = - Q @ x_bar[-1,:]
    p = q
    K_gains = np.empty((N, R_list[0].shape[0], Q_list[0].shape[0]))
    k_feedforward = np.empty((N, R_list[0].shape[0]))
    for i in range(N):
        A = A_list[N-1-i]
        B = B_list[N-1-i]
        Q = Q_list[N-1-i]
        R = R_list[N-1-i]

        T = B.T @ P @ B + R
        K = - np.linalg.inv(T) @ B.T @ P @ A
        k = - np.linalg.inv(T) @ B.T @ p
        q = - Q @ x_bar[N-1-i, :]
        p = q + A.T @ p + A.T @ P @ B @ k
        P = Q + A.T @ P @ A + A.T @ P @ B @ K
        k_feedforward[N-1-i] = k
        K_gains[N-1-i] = K

    return K_gains, k_feedforward

def get_torque_to_track_traj_const_ori(p_d, p_d_dt, p_d_dtdt, R_d, Kp, Kd, Minv, J, dJdq, dq, p, R):
    """
    p_d: desired position, shape (3,)
    p_d_dt: desired translation velocity, shape (3,)
    p_d_dtdt: desired translation acceleration, shape (3,)
    R_d: desired orientation, shape (3,3)
    Kp: proportional gain, shape (6,6)
    Kd: derivative gain, shape (6,6)
    Minv: inverse of mass matrix, shape (n_joints,n_joints)
    J: Jacobian, shape (6,n_joints)
    dJdq: dJdq, shape (6,)
    dq: current joint velocities, shape (n_joints,)
    p: current position, shape (3,)
    R: current orientation, shape (3,3)
    """

    J_dq = np.dot(J, dq)
    v = J_dq[:3]
    omega = J_dq[3:]

    # Translation errors
    e_p = p_d - p
    e_p_dt = p_d_dt - v

    # Quaternion errors
    quat_d = get_quat_from_rot_matrix(R_d)
    qv_d = quat_d[:3]
    qw_d = quat_d[3]
    quat = get_quat_from_rot_matrix(R)
    qv = quat[:3]
    qw = quat[3]
    S_qv_d = get_skew_symmetric_matrix(qv_d)
    S_qv = get_skew_symmetric_matrix(qv)
    qw_dt = - 0.5 * qv @ omega
    I_3 = np.eye(3).astype(config.np_dtype)
    qv_dt = 0.5 * (qw * I_3 - S_qv) @ omega

    e_o = qw*qv_d - qw_d*qv - S_qv_d @ qv
    e_o_dt = qw_dt*qv_d - qw_d*qv_dt - S_qv_d @ qv_dt
    
    # All errors
    e = np.zeros(6, dtype=config.np_dtype)
    e[:3] = e_p
    e[3:] = e_o

    e_dt = np.zeros(6, dtype=config.np_dtype)
    e_dt[:3] = e_p_dt
    e_dt[3:] = e_o_dt

    # Control law
    S_qv_dt = get_skew_symmetric_matrix(qv_dt)
    feed_forward = np.zeros(6, dtype=config.np_dtype)
    feed_forward[:3] = p_d_dtdt
    feed_forward[3:] = -0.5 * qv_d * (qv_dt @ omega) - 0.5 * (qw_d * I_3 + S_qv_d) @ (qw_dt * I_3 - S_qv_dt) @ omega
    D = np.zeros((6,6), dtype=config.np_dtype)
    D[:3,:3] = - I_3
    D[3:,3:] = - 0.5 * np.outer(qv_d, qv) - 0.5 * (qw_d * I_3 + S_qv_d) @ (qw * I_3 - S_qv)

    G = D @ J @ Minv # shape (6,n_joints)
    u_task = -feed_forward - Kp @ e - Kd @ e_dt - D @ dJdq

    return G, u_task

