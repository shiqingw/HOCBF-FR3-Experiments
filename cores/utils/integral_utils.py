from scipy.integrate import dblquad
import numpy as np

# Define the function to be integrated
h = 0.05
w = 0.125
func = lambda y, x: np.sqrt(x**2 + y**2)/(h*w)

# Perform the numerical integration
integral_result, error = dblquad(func, -w/2, w/2, lambda x: -h/2, lambda x: h/2)

print(integral_result)
print(error)