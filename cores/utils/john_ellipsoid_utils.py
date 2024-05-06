import numpy as np
import cvxpy as cp

def solve_john_ellipsoids(x, threshold=1e-6):
    """
    Solve the John Ellipsoid problem using CVXPY
    x: the input data, shape (num_points, self.dim)
    """

    dim = x.shape[1]
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

    return Q_np, c_np

h = 0.125
w = 0.05
points = np.array([[h/2, w/2],
                  [h/2, -w/2],
                  [-h/2, w/2],
                  [-h/2, -w/2]])
Q, c = solve_john_ellipsoids(points)
print(Q)
print(np.sqrt(np.linalg.inv(Q)))