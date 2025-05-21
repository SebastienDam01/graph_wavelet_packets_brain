"""
Functions that constructs the dual graph of a graph.

Sébastien Dam
"""
import numpy as np

def dualGraph(eigvect, Q, edge_weight):
    """
    Construct a dual graph based on distances between eigenvectors computed from the DAG pseudometric.

    Parameters:
    ----------
    eigvect : np.ndarray
        Eigenvectors.
    Q : np.ndarray
        Incidence matrix.
    edge_weight : np.ndarray
        Edge weights.

    Returns:
    -------
    W_star: np.ndarray
        Dual graph.
    """
    N = len(eigvect)
    distDAG = eigDAG_Distance(eigvect, Q, edge_weight=edge_weight)
    W_star = np.zeros((N, N))
    for i in range(0, N - 1):
        for j in range(i+1, N):
            W_star[i, j] = 1 / distDAG[i, j]
            
    W_star = W_star + W_star.T
    
    return W_star