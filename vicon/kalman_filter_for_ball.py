import numpy as np
import scipy.sparse as sparse

class KalmanFilter:
    def __init__(self, dt, Q, R, x0, P0):

        # Check shape
        assert Q.shape == (6,6)
        assert R.shape == (3,3)
        assert x0.shape == (6,)
        assert P0.shape == (6,6)

        # Define dynamics
        A = np.eye(6)
        A[0:3, 3:6] = np.eye(3) * dt
        self.A = sparse.csc_matrix(A)

        B = np.zeros((6, 3))
        B[3:6, 0:3] = np.eye(3) * dt
        self.B = sparse.csc_matrix(B)

        C = np.zeros((3,6))
        C[0:3, 0:3] = np.eye(3)
        self.C = sparse.csc_matrix(C)

        # Uncertainty and initial conditions
        self.Q = sparse.csc_matrix(Q*dt)
        self.R = sparse.csc_matrix(R)
        self.x = x0
        self.P = sparse.csc_matrix(P0)
        self.g = np.array([0, 0, -9.81])
    
    def predict(self):
        self.x = self.A @ self.x + self.B @ self.g
        self.P = self.A @ self.P @ self.A.T + self.Q
    
    def update(self, z):
        y = z - self.C @ self.x
        S = self.C @ self.P @ self.C.T + self.R
        K = self.P @ self.C.T @ sparse.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = self.P - K @ self.C @ self.P
    
    def get_state(self):
        return self.x