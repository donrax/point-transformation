import warnings
import numpy as np

def apply_transformation(X, T):
    """
    Applies the 2D or 3D transformation matrix T to 2D or 3D data points in X.
    :param X: Nx2 or Nx3 numpy ndarray
    :param T: 2D (3x3) or 3D (4x4) numpy ndarray
    :return: Transformed points
    """

    # Check input
    if not isinstance(X, np.ndarray):
        warnings.warn("Input matrix X must be of type numpy.ndarray! X is of type " + str(type(X)))
    if not isinstance(T, np.ndarray):
        warnings.warn("Transformation matrix T must be of type numpy.ndarray! T is of type "+str(type(T)))
    if X.shape[1] != 2L and X.shape[1] != 3L:
        warnings.warn("Input matrix X must be of shape Nx2 or Nx3! X is of shape " + str(X.shape))
    if T.shape != (3L, 3L) and T.shape != (4L, 4L):
        warnings.warn("Transformation matrix must be of shape  3x3 (2D) or 4x4 (3D)! T is of shape " + str(T.shape))
    if not (X.shape[1] == 2L and T.shape == (3L, 3L)) and not (X.shape[1] == 3L and T.shape == (4L, 4L)):
        warnings.warn("Data points in X and transformation matrix T must be of correct dimensions!" + ' X is '+ str(X.shape) + ' T is '+ str(T.shape))

    # Add homogeneus coordinate
    X = np.vstack([X.T, np.ones(X.shape[0])])
    # Transform points
    t_X = T.dot(X).T
    # Divide by homogeneus value
    t_X = np.asarray([row/row[-1] for row in t_X])

    # Return without homogeneus coordinate
    return t_X[:,0:-1]