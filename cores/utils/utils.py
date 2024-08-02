import numpy as np
import pickle
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster

def seed_everything(seed: int = 0):
    np.random.seed(seed)
    # torch.manual_seed(seed)
    # if torch.cuda.is_available(): 
    #     torch.cuda.manual_seed(seed)
    #     torch.cuda.manual_seed_all(seed)
    #     torch.backends.cudnn.deterministic = True
    #     torch.backends.cudnn.benchmark = False
    # if torch.backends.mps.is_available():
    #     torch.mps.manual_seed(seed)

def save_dict(dict_obj, fullname):
    with open(fullname, 'wb') as handle:
        pickle.dump(dict_obj, handle, protocol=pickle.HIGHEST_PROTOCOL)

def load_dict(fullname):
    with open(fullname, 'rb') as handle:
        loaded_obj = pickle.load(handle)
    return loaded_obj

def dict2func(d):
    return lambda x: d[x]

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f

def points2d_to_ineq(corners):
    """
    corners: np.array of shape (n, 2), arranged in CLOCKWISE order
    """
    n = corners.shape[0]
    A = np.empty((n, 2))
    b = np.empty(n)
    for i in range(n):
        A[i, 0] = - corners[(i+1)%n, 1] + corners[i, 1]
        A[i, 1] = - corners[i, 0] + corners[(i+1)%n, 0]
        b[i] = -corners[i, 1]*corners[(i+1)%n, 0] + corners[(i+1)%n, 1]*corners[i, 0]
    norm_A = np.linalg.norm(A, axis=1)
    A = A / norm_A[:, None]
    b = b / norm_A
    return A, b

def get_skew_symmetric_matrix(v):
    """
    v: np.array of shape (3,)
    """
    return np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])

def get_orthogonal_vector(v, u):
    v_dot_v = np.dot(v, v)
    if v_dot_v == 0:
        raise ValueError("Vector v must not be the zero vector.")
    proj_u_on_v = (np.dot(u, v) / v_dot_v) * v  # Project u onto v
    w = u - proj_u_on_v  # Subtract to get the orthogonal component
    # Normalize w
    if np.linalg.norm(w) != 0:
        w = w / np.linalg.norm(w)
    return w

def unique_rows_with_tolerance(arr, tol):
    # Compute the pairwise distances between rows
    pairwise_dist = pdist(arr)
    # Convert the pairwise distances to a square form distance matrix
    dist_matrix = squareform(pairwise_dist)
    # Perform hierarchical clustering
    Z = linkage(dist_matrix, method='single')
    # Form flat clusters based on the given tolerance
    cluster_labels = fcluster(Z, tol, criterion='distance')
    # Find the unique clusters and their representative rows
    unique_clusters = np.unique(cluster_labels)
    unique_rows = np.array([arr[cluster_labels == cluster][0] for cluster in unique_clusters])
    
    return unique_rows

def get_facial_equations(vertices):
    """
    vertices: np.array of shape (n_vertices, dim)
    
    Returns:
    The matrices A and b such that A * x + b <= 0 defines the polytope
    """
    hull = ConvexHull(vertices)
    dim = vertices.shape[1]
    tmp = np.zeros((len(hull.simplices), dim+1))

    for (i, simplex) in enumerate(hull.simplices):
        facial_vertices = vertices[simplex]
        homogeneous_coordinates = np.c_[facial_vertices, np.ones(facial_vertices.shape[0])]
        coeffs = np.linalg.svd(homogeneous_coordinates)[-1][-1, :]
        coeffs[np.abs(coeffs) < 1e-6] = 0
        tmp[i] = coeffs
    
    # Adjust the signs of the equations
    mean_point = np.mean(vertices, axis=0)
    for i in range(len(tmp)):
        if np.dot(tmp[i,0:dim], mean_point) + tmp[i,dim] > 0:
            tmp[i] = -tmp[i]

    # normalize to have a unit norm
    norm = np.linalg.norm(tmp[:,0:dim], axis=1)
    tmp = tmp / norm[:, np.newaxis]
    
    tmp = unique_rows_with_tolerance(tmp, 1e-5)
    A = tmp[:,0:dim]
    b = tmp[:,dim]

    return A, b
