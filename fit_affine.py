import numpy as np
from normalize_points import normalize_points

def fit_affine(src, dst, weg=np.array([])):
    """
    Fits an affine transformation between two d-dimensional point sets.
    :param src: Nxd numpy ndarray d-dimensional data points
    :param dst: Nxd numpy ndarray d-dimensional data points
    :param weg: relative weights of data points (Nx1 numpy ndarray)
    :return: transformation matrix T from src to dst
    """

    # -------------------------------
    # Normalize points
    # -------------------------------
    src, Ts = normalize_points(src)
    dst, Td = normalize_points(dst)

    # -------------------------------
    # Set parameters
    # -------------------------------
    # Get number of d-dimensional points
    n = len(src)
    # Check if weights are provided
    if not weg.any():
        weg = np.ones(n)

    # -------------------------------
    # Construct matrix A and vector b
    # -------------------------------
    A = np.zeros([2*n,6])
    b = np.zeros([2*n,1])
    W = np.zeros([2*n,2*n])
    for i in xrange(n):
        # Construct matrix A
        # ax = (x, y, 1, 0, 0, 0)
        # ay = (0, 0, 0, x, y, 1)
        A[i*2  ,:] = np.array([ src[i,0], src[i,1], 1, 0, 0, 0 ])
        A[i*2+1,:] = np.array([ 0, 0, 0, src[i,0], src[i,1], 1 ])
        # Construct vector b
        b[i*2  ] = dst[i,0]
        b[i*2+1] = dst[i,1]
        # Construct diagonal weigh matrix
        W[i*2  ,i*2  ] = weg[i]
        W[i*2+1,i*2+1] = weg[i]

    # -------------------------------
    # Estimate solution x
    # -------------------------------
    x = np.linalg.solve( (A.T.dot(W)).dot(A) , (A.T.dot(W)).dot(b) )
    T = np.vstack([np.reshape(x,(2,3)) ,[0,0,1]])

    # -------------------------------
    # Denormalize transformation
    # -------------------------------
    T, _, _, _ = np.linalg.lstsq(Td,T.dot(Ts))

    return T