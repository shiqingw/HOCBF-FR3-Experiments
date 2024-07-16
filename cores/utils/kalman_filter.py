import numpy as np
import scipy.sparse as sparse

class KalmanFilter:
    def __init__(self, dt, Q, R, x0, P0):
        # Check if the matrices have the correct dimensions
        assert Q.shape == (14, 14)
        assert R.shape == (14, 14)
        assert x0.shape == (14,)
        assert P0.shape == (14, 14)

        # Dynamics of q and dq
        # q_{t+1} = q_t + dq_t * dt
        # dq_{t+1} = dq_t + ddq_t * dt
        # ddq_t is taken as the control input

        A = np.eye(14)
        A[0:7, 7:14] = np.eye(7)*dt
        self.A = sparse.csc_matrix(A)

        B = np.zeros((14, 7))
        B[7:14, 0:7] = np.eye(7)*dt
        self.B = sparse.csc_matrix(B)

        C = np.eye(14)
        self.C = sparse.csc_matrix(C)
        
        self.Q = sparse.csc_matrix(Q)
        self.R = sparse.csc_matrix(R)
        self.x = x0
        self.P = sparse.csc_matrix(P0)

    def predict(self, u):
        self.x = self.A @ self.x + self.B @ u
        self.P = self.A @ self.P @ self.A.T + self.Q

    def update(self, z):
        y = z - self.C @ self.x
        S = self.C @ self.P @ self.C.T + self.R
        K = self.P @ self.C.T @ sparse.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = self.P - K @ self.C @ self.P

    def estimate(self):
        return self.x
    