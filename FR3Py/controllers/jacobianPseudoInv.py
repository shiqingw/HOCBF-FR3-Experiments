import numpy as np
import numpy.linalg as LA
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Rotation as R

from FR3Py.robot.model import PinocchioModel
from FR3Py.controllers.solvers import QPSolver


def axis_angle_from_rot_mat(rot_mat):
    rotation = R.from_matrix(rot_mat)
    axis_angle = rotation.as_rotvec()
    angle = LA.norm(axis_angle)
    axis = axis_angle / angle
    return axis, angle


def get_R_end_from_start(x_ang, y_ang, z_ang, R_start):
    """Get target orientation based on initial orientation"""
    _R_end = (
        R.from_euler("x", x_ang, degrees=True).as_matrix()
        @ R.from_euler("y", y_ang, degrees=True).as_matrix()
        @ R.from_euler("z", z_ang, degrees=True).as_matrix()
        @ R_start
    )
    R_end = R.from_matrix(_R_end).as_matrix()
    return R_end


class WaypointController:
    def __init__(self, kp=0.1):
        # define solver
        self.robot = PinocchioModel()
        self.solver = QPSolver(9)
        self.initialized = False
        self.kp = kp

    def compute(self, q, dq, T_cmd=None):
        # Get the robot paramters for the given state
        info = self.robot.getInfo(q, dq)
        if T_cmd is not None:
            self.p_cmd = T_cmd[0:3, -1].reshape(3, 1)
            self.R_cmd = T_cmd[0:3, 0:3]

        if not self.initialized:
            # get initial rotation and position
            self.R_start, _p_start = info["R_EE"], info["P_EE"]
            self.p_start = _p_start[:, np.newaxis]
            if T_cmd is None:
                # get target rotation and position
                self.p_cmd = np.array([[0.5], [0], [0.5]])
                self.R_cmd = get_R_end_from_start(0, 0, 0, self.R_start)

            # compute R_error, ω_error, θ_error
            self.R_error = self.R_cmd @ self.R_start.T
            self.ω_error, self.θ_error = axis_angle_from_rot_mat(self.R_error)
            self.initialized = True

        # get end-effector position
        p_current = info["P_EE"][:, np.newaxis]

        # get end-effector orientation
        R_current = info["R_EE"]

        # get Jacobians from info
        pinv_jac = info["pJ_HAND"]
        jacobian = info["J_HAND"]

        # compute joint-centering joint acceleration
        dq_nominal = 0.5 * (self.robot.q_nominal - q[:, np.newaxis])

        # compute error rotation matrix
        R_err = self.R_cmd @ R_current.T

        # compute orientation error in axis-angle form
        rotvec_err = Rotation.from_matrix(R_err).as_rotvec()

        # compute EE position error
        p_error = np.zeros((6, 1))
        p_error[:3] = self.p_cmd - p_current
        p_error[3:] = rotvec_err[:, np.newaxis]

        # compute EE velocity target
        dp_target = np.zeros((6, 1))
        params = {
            "Jacobian": jacobian,
            "p_error": p_error,
            "p_current": p_current,
            "dp_target": dp_target,
            "Kp": self.kp * np.eye(6),
            "dq_nominal": dq_nominal,
            "nullspace_proj": np.eye(9) - pinv_jac @ jacobian,
        }

        # solver for target joint velocity
        self.solver.solve(params)
        dq_target = self.solver.qp.results.x
        return dq_target
