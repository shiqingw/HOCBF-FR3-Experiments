import cvxpy as cp
import numpy as np
from scipy.spatial.transform import Rotation
from cores.configuration.configuration import Configuration
config = Configuration()

class JohnEllipsoidCollection:
    def __init__(self, dim, n_obsctacles, obs_dict):
        self.dim = dim
        self.n_obsctacles = n_obsctacles
        self.centers = np.zeros((n_obsctacles, dim))
        self.Qs = np.zeros((n_obsctacles, dim, dim))
        self.max_radii = np.zeros(n_obsctacles)
        self.sizes = np.zeros((n_obsctacles, dim))
        self.Rs = np.zeros((n_obsctacles, dim, dim))
        
        for (i, key) in enumerate(obs_dict.keys()):
            obs = obs_dict[key]
            if obs["type"] == "ellipsoid":
                size = np.array(obs["size"], dtype=config.np_dtype)
                quat = np.array(obs["quat"], dtype=config.np_dtype)
                c = np.array(obs["pos"], dtype=config.np_dtype)
                max_radius = c.max()
                D = np.diag(1/size**2)
                R = Rotation.from_quat(quat).as_matrix()
                Q = R @ D @ R.T
            elif obs["type"] == "cylinder":
                if self.dim != 3:
                    raise ValueError("Cylinder obstacles are only supported in 3D")
                r, h = np.array(obs["size"], dtype=config.np_dtype)
                c = np.array(obs["pos"], dtype=config.np_dtype)
                quat = np.array(obs["quat"], dtype=config.np_dtype)
                D = np.diag([2/(3*r**2), 2/(3*r**2), 1/(3*h**2)]).astype(config.np_dtype)
                R = Rotation.from_quat(quat).as_matrix()
                Q = R @ D @ R.T
                max_radius = np.sqrt(max(3*r**2/2, 3*h**2))
            else:
                points = self.find_points(obs)
                Q, c, max_radius = self.solve_john_ellipsoids(points)
            self.centers[i] = c
            self.Qs[i] = Q
            self.max_radii[i] = max_radius

            # spectral decomposition of Q = R D R^T
            D, R = np.linalg.eigh(Q)
            self.sizes[i] = 1/np.sqrt(D)
            self.Rs[i] = R

    def find_points(self, obs):
        if obs["type"] == "box":
            if self.dim != 3:
                    raise ValueError("Box obstacles are only supported in 3D")
            size = np.array(obs["size"], dtype=config.np_dtype)
            c = np.array(obs["pos"], dtype=config.np_dtype)
            quat = np.array(obs["quat"], dtype=config.np_dtype)
            quat = quat / np.linalg.norm(quat)
            points = np.array([[1, 1, 1],
                                 [1, 1, -1],
                                 [1, -1, 1],
                                 [1, -1, -1],
                                 [-1, 1, 1],
                                 [-1, 1, -1],
                                 [-1, -1, 1],
                                 [-1, -1, -1]], dtype=config.np_dtype)
            points = points * size
            R = Rotation.from_quat(quat).as_matrix()
            points = points @ R.T + c
        else:
            raise ValueError("Unsupported obstacle type")
        return points

    def solve_john_ellipsoids(self, x, threshold=1e-6):
        """
        Solve the John Ellipsoid problem using CVXPY
        x: the input data, shape (num_points, self.dim)
        """

        dim = self.dim
        num_points = x.shape[0]

        # Define the variable for A which should be symmetric positive semidefinite
        A = cp.Variable((dim, dim), PSD=True)

        # Define the variable for b
        b = cp.Variable(dim)

        # The constraints are that the 2-norm of (A*xi + b) is less than or equal to 1
        constraints = [cp.norm(A @ x[i] + b, 2) <= 1 for i in range(num_points)]

        # The objective is to minimize the negative log determinant of A
        objective = cp.Minimize(-cp.log_det(A))

        # Define the problem and solve it
        problem = cp.Problem(objective, constraints)
        problem.solve()

        # After solving the problem, A.value and b.value will contain the optimized values
        A_np = A.value
        b_np = b.value

        # Smooth out values that are close to zero
        A_np[np.abs(A_np) < threshold] = 0
        b_np[np.abs(b_np) < threshold] = 0

        # Convert to (x-c)^T Q (x-c) <= 1 form
        Q_np = A_np @ A_np
        c_np = - np.linalg.solve(A_np, b_np)
        max_radius = 1 / np.linalg.eigvals(A_np).min()

        return Q_np, c_np, max_radius
    
    def find_ellipsoids_given_threshold(self, point, threshold=1):
        """
        Find the closest obstacle to a given point
        point: the point to find the closest obstacle to
        threshold: the maximum distance to consider an obstacle
        """
        distances = np.linalg.norm(self.centers - point, axis=1) - self.max_radii
        closest_obstacles = np.where(distances < threshold)
        Qs = self.Qs[closest_obstacles]
        centers = self.centers[closest_obstacles]
        
        return distances[closest_obstacles], Qs, centers
    
    def find_k_closest_ellipsoids(self, point, k=1):
        """
        Find the k closest obstacles to a given point
        point: the point to find the closest obstacle to
        k: the number of obstacles to find
        """
        if k > self.n_obsctacles:
            k = self.n_obsctacles
        distances = np.linalg.norm(self.centers - point, axis=1) - self.max_radii
        closest_obstacles = np.argsort(distances)[:k]
        Qs = self.Qs[closest_obstacles]
        centers = self.centers[closest_obstacles]

        return distances[closest_obstacles], Qs, centers