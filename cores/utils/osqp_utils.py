import osqp
import scipy
import numpy as np
from cores.configuration.configuration import Configuration
config = Configuration()

def init_osqp(n_v, n_in):
    P = scipy.sparse.eye(n_v, format='csc', dtype=config.np_dtype) 

    data = np.zeros(n_v*n_in, dtype=config.np_dtype)
    row_indices = np.tile(np.arange(n_in), n_v)
    col_ptrs = np.arange(0, n_in*n_v+1, n_in)
    A = scipy.sparse.csc_matrix((data, row_indices, col_ptrs), shape=(n_in, n_v))

    l = np.full(n_in, -np.inf)
    u = np.full(n_in, np.inf)
    q = np.zeros(n_v, dtype=config.np_dtype)

    qp = osqp.OSQP()
    qp.setup(P=P, q=q, A=A, l=l, u=u, verbose=False)
    _ = qp.solve()

    return qp