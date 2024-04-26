import numpy as np
import diffOptHelper as doh

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

print(doh.getDualVariable(a, b))
