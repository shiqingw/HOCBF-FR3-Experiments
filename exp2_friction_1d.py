import numpy as np
import matplotlib.pyplot as plt

def fric_fun(Fz_abs, v, mu_coulomb, mu_static):
    v_norm = np.abs(v) 
    v_dir = np.sign(v)
    F_coulomb = mu_coulomb * Fz_abs * np.tanh(100*v_norm) * v_dir
    F_static = mu_static * Fz_abs * v_norm * np.exp(-50*v_norm) * v_dir
    return F_coulomb + F_static

v_range = np.linspace(-1, 1, 400)

Fz_abs = 10
mu_coulomb = 0.305
mu_static = 100

F_list = np.zeros_like(v_range)
for i in range(len(v_range)):
    F_list[i] = fric_fun(Fz_abs, v_range[i], mu_coulomb, mu_static)

fig = plt.figure()
plt.plot(v_range, F_list)
plt.show()