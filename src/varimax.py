"""
Functions that implements varimax rotation.

Sébastien Dam
"""
import numpy as np 

def varimax(A, gamma=1, minit=20, maxit=1000, tol=1e-12):
    """
    Varimax rotation.

    Parameters
    ----------
    A : np.ndarray
        input matrix, whose column vectors are to be rotated. d, m = size(A)..
    gamma : float, optional
        help to define a good initial orthogonal matrix. The default is 1.
    minit : int, optional
        Minimum number of iterations, in case of the stopping criteria fails initially.. The default is 20.
    maxit : int, optional
        Maximum number of iterations. The default is 1000.
    tol : float, optional
        Relative tolerance for stopping criteria.. The default is 1e-12.

    Returns
    -------
    TYPE
        DESCRIPTION.
        
    References
    ----------
    J. IRION, H. LI, N. SAITO, AND Y. SHAO, MultiscaleGraphSignalTransforms.jl. https://github.com/UCD4IDS/MultiscaleGraphSignalTransforms.jl, 2021.
    https://github.com/UCD4IDS/MultiscaleGraphSignalTransforms.jl/blob/master/src/varimax.jl

    """
    
    # Get the sizes of input matrix
    d, m = A.shape
    # If there is only one vector, then do nothing.
    if m == 1:
        return A
    if d == m and np.linalg.matrix_rank(A) == d:
        return np.eye(d, m)
    # Warm up step: start with a good initial orthogonal matrix T by SVD and QR
    T = np.eye(m)
    B = A.dot(T)
    L, _, M = np.linalg.svd(A.T.dot(d * B**3 - gamma * B.dot(np.diag(np.sum(B**2, axis=0)))))
    T = L.dot(M.T)
    if np.linalg.norm(T - np.eye(m)) < tol:
        T = np.linalg.qr(np.random.randn(m, m))[0]
        B = A.dot(T)
        
    # Iteration step: get better T to maximize the objective (as described in Factor Analysis book)
    D = 0
    for k in range(maxit):
        Dold = D
        L, s, M = np.linalg.svd(A.T.dot(d * B**3 - gamma * B.dot(np.diag(np.sum(B**2, axis=0)))))
        T = L.dot(M.T)
        D = np.sum(s)
        B = A.dot(T)
        if (np.abs(D - Dold) / D < tol) and k >= minit:
            break
        
    # Adjust the sign of each rotated vector such that the maximum absolute value is positive.
    for i in range(m):
        if np.abs(np.max(B[:, i])) < np.abs(np.min(B[:, i])):
            B[:, i] = -B[:, i]
    
    return B
