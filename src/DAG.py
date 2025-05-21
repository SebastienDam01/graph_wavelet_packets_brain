"""
Functions to compute the DAG pseudometric between eigenvectors of a matrix.

Sébastien Dam
"""
import numpy as np


def dag_pseudometric(Q, u, v):
    return np.linalg.norm(np.abs(Q.T @ u) - np.abs(Q.T @ v))

def eigDAG_Distance(𝚽, Q, edge_weight, numEigs):
    """
    # Input Arguments
    - `𝚽::Matrix{Float64}`: matrix of graph Laplacian eigenvectors, 𝜙ⱼ₋₁ (j = 1,...,size(𝚽,1)).
    - `Q::Matrix{Float64}`: incidence matrix of the graph.
    - `numEigs::Int64`: number of eigenvectors considered.
    - `edge_weight::Any`: default value is 1, stands for unweighted graph
        (i.e., all edge weights equal to 1). For weighted graph, edge_weight is the
        weights vector, which stores the affinity weight of each edge.
    
    # Output Argument
    - `dis::Matrix{Float64}`: a numEigs x numEigs distance matrix, dis[i,j] = d_DAG(𝜙ᵢ₋₁, 𝜙ⱼ₋₁).
    """
    dis = np.zeros((numEigs, numEigs))
    abs_𝚽 = np.abs(Q.T @ 𝚽)
    for i in range(numEigs):
        for j in range(i+1, numEigs):
            dis[i,j] = np.linalg.norm((abs_𝚽[:,i] - abs_𝚽[:,j]) * np.sqrt(edge_weight), 2)
    return dis + dis.T