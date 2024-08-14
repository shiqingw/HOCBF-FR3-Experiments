import numpy as np

def find_coefficients(t_samples, p_samples):
    """
    Given N samples of t and corresponding p(t), return the coefficients c_2, c_1, c_0
    where p(t) = 1/2 * c_2 * t^2 + c_1 * t + c_0.
    
    Parameters:
    t_samples (numpy.ndarray): Array of time samples (shape: (N,))
    p_samples (numpy.ndarray): Array of p(t) samples (shape: (N, 3))
    
    Returns:
    c2 (numpy.ndarray): Coefficient vector c_2 (shape: (3,))
    c1 (numpy.ndarray): Coefficient vector c_1 (shape: (3,))
    c0 (numpy.ndarray): Coefficient vector c_0 (shape: (3,))
    """
    # Construct the design matrix A
    A = np.vstack([0.5 * t_samples**2, t_samples, np.ones_like(t_samples)]).T
    
    # Solve for the coefficients using the least squares method
    coefficients, _, _, _ = np.linalg.lstsq(A, p_samples, rcond=None)
    
    c2, c1, c0 = coefficients
    
    return c2, c1, c0

# Example usage:
t_samples = np.array([0, 1, 2, 3])
p_samples = np.array([
    [0, 0, 0],
    [0.5, 0.5, 0.5],
    [2, 2, 2],
    [4.5, 4.5, 4.5]
])

c2, c1, c0 = find_coefficients(t_samples, p_samples)
print("c2:", c2)
print("c1:", c1)
print("c0:", c0)