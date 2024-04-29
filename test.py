import mujoco
import numpy as np
from FR3Py.sim.mujoco_with_contact import FR3Sim
from FR3Py import ASSETS_PATH

# Mujoco simulation
mj_env = FR3Sim(xml_path=os.path.join(ASSETS_PATH, "mujoco/fr3.xml"))
mj_env.reset(np.array(initial_joint_angles, dtype = config.np_dtype))
mj_env.step()
dt = mj_env.dt