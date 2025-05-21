"""
Functions that implements partitioning of the eigenvectors of a graph using the Fiedler vector.

Sébastien Dam
"""
import numpy as np
import copy
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import eigs, svds

def partition_fiedler_pm(v):
    """
    Convert a Fiedler vector into a binary partition vector (+1 or -1).

    Parameters:
    ----------
    v : np.ndarray
        The Fiedler vector.

    Returns:
    -------
    pm : np.ndarray
        Binary partition vector with values in {+1, -1}.
    v : np.ndarray
        Modified Fiedler vector if applicable (first significant entry made positive).
    """
    tol = 1e3 * np.finfo(float).eps
    
    # set the first nonzero element to be positive
    row = 0
    while abs(v[row]) < tol and row < len(v):
        row += 1
    if v[row] < 0:
        v = -v
        
    # assign each point to either region 1 or region -1, and assign any zero entries to the smaller region
    if np.sum(v >= tol) > np.sum(v <= -tol):
        pm = 2 * (v >= tol) - 1
    else:
        pm = 2 * (v <= -tol) - 1
        
    # make sure the first point is assigned to region 1 (not -1)
    if pm[0] < 0:
        pm = -pm
    return pm, v

def partition_fiedler_troubleshooting(pm, v, W, val):
    """
   Troubleshoot and fix partition issues where all nodes are in one cluster or have ambiguous assignment.

   Parameters:
   ----------
   pm : np.ndarray
       Initial partition vector.
   v : np.ndarray
       Fiedler vector.
   W : np.ndarray
       Adjacency matrix of the graph.
   val : float
       Algebraic connectivity (second smallest eigenvalue of the Laplacian).

   Returns:
   -------
   pm : np.ndarray
       Corrected partition vector.
   """
    # Initial setup
    N = len(pm)
    tol = 1e3 * np.finfo(float).eps
    
    # If pm vector indicates no partition (single sign) or ambiguities via
    # 0 entries, then we'll trouble shoot.
    if np.sum(pm < 0) == 0 or np.sum(pm > 0) == 0 or np.sum(np.abs(pm)) < N:
        # Case 1: an input graph is not connected
        if val < tol:
            pm = 2 * (np.abs(v) > tol) - 1
            while np.sum(pm < 0) == 0 or np.sum(pm > 0) == 0:
                tol *= 10
                pm = 2 * (np.abs(v) > tol) - 1
                if tol > 1:
                    pm[:int(np.ceil(N / 2))] = 1
                    pm[int(np.ceil(N / 2)):] = -1
        # Case 2: it is connected
        else:
            pm = 2 * (v >= np.mean(v)) - 1
            if np.sum(np.abs(pm)) < N:
                pm0 = np.where(pm == 0)[0]
                pm[pm0] = (W[pm0, :].dot(v) > tol) - (W[pm0, :].dot(v) < -tol)
                pm[np.where(pm == 0)[0]] = (np.sum(pm > 0) - np.sum(pm < 0)) >= 0
                
        # if one region has no points after all the above processing
        if np.sum(pm < 0) == 0 or np.sum(pm > 0) == 0:
            pm[:int(np.ceil(N / 2))] = 1
            pm[int(np.ceil(N / 2)):] = -1
        
    # make sure that the first point is assigned as a 1
    if pm[0] < 0:
        pm = -pm
    return pm

def partition_fiedler_troubleshooting_pm(pm):
    """
    Fix a partition vector if it fails to divide nodes into two non-empty groups.

    Parameters:
    ----------
    pm : np.ndarray
        Binary partition vector.

    Returns:
    -------
    pm : np.ndarray
        Corrected partition vector with both +1 and -1, and first element as +1.
    """
    N = len(pm)
    
    if np.sum(pm < 0) == 0 or np.sum(pm > 0) == 0:
        pm[:int(np.ceil(N / 2))] = 1
        pm[int(np.ceil(N / 2)):] = -1
        
    # make sure that the first point is assigned as a 1
    if pm[0] < 0:
        pm = -pm
    return pm

def partition_fiedler(W, method='L', v=None):
    """
    Perform graph partitioning using the Fiedler vector.

    Parameters:
    ----------
    W : np.ndarray
        Weighted adjacency matrix of the graph.
    method : str, optional
        Type of Laplacian to use: 'L' (unnormalized) or 'Lrw' (random-walk normalized). Default is 'L'.
    v : np.ndarray, optional
        Precomputed Fiedler vector. If provided, avoids eigenvalue computation.

    Returns:
    -------
    pm : np.ndarray
        Partition vector with values in {+1, -1}.
    v : np.ndarray
        Fiedler vector used for partitioning.
    """
    if v is not None:
        pm, v = partition_fiedler_pm(v)
        if W.shape[0] == len(v):
            val = v.dot((np.diag(np.sum(W, axis=0)) - W).dot(v))
            pm = partition_fiedler_troubleshooting(pm, v, W, val)
        else:
            pm = partition_fiedler_troubleshooting_pm(pm)
        return pm, v

    N = W.shape[0]
    eigs_flag = 0
    sigma = np.finfo(float).eps
    cutoff = 128
    if N == 2:
        pm = np.array([1, -1])
        v = pm / np.sqrt(2)
        return pm, v

    if method == 'L' or np.min(np.sum(W, axis=0)) < 1e3 * sigma:
        if N > cutoff:
            v0 = np.ones(N) / np.sqrt(N)
            try:
                val, vtmp = eigs(csc_matrix(np.diag(np.sum(W, axis=0)) - W),
                                 k=2, sigma=sigma, v0=v0)
                if np.iscomplex(val[0]):
                    eigs_flag = 1
            except Exception as emsg:
                print("Exception in eigs(L) occurred: ", emsg)
                eigs_flag = 2
            if eigs_flag == 0:
                val, ind = np.max(val), np.argmax(val)
                v = vtmp[:, ind]
        if N <= cutoff or eigs_flag != 0:
            colsumW = np.sum(W, axis=0)
            D = np.diag(colsumW)
            D2 = np.diag(colsumW**(-0.5))
            vtmp, val, _ = svds(D2.dot(D - W).dot(D2))
            val = val[-2]
            v = vtmp[:, -2]
            v = (np.sum(W, axis=1)**(-0.5)) * v
    elif method == 'Lrw':
        if N > cutoff:
            v0 = np.ones(N) / np.sqrt(N)
            try:
                temp = csc_matrix(np.diag(np.sum(W, axis=0)))
                val, vtmp = eigs(temp - W, temp, k=2, sigma=sigma, v0=v0)
                if np.iscomplex(val[0]):
                    eigs_flag = 1
            except Exception as emsg:
                print("Exception in eigs(Lrw) occurred: ", emsg)
                eigs_flag = 2
            if eigs_flag == 0:
                val, ind = np.max(val), np.argmax(val)
                v = vtmp[:, ind]
        if N <= cutoff or eigs_flag != 0:
            colsumW = np.sum(W, axis=0)
            D = np.diag(colsumW)
            D2 = np.diag(colsumW**(-0.5))
            vtmp, val, _ = svds(D2.dot(D - W).dot(D2))
            val = val[-2]
            v = vtmp[:, -2]
            v = (np.sum(W, axis=1)**(-0.5)) * v
    else:
        raise ValueError("Graph partitioning method: {} is not recognized!".format(method))

    pm, v = partition_fiedler_pm(v)
    pm = partition_fiedler_troubleshooting(pm, v, W, val)
    return pm, v

def partition_tree_fiedler(G, method='Lrw', swapRegion=True, jmax=None):
    """
    Recursively partition a graph into a binary tree of graph wavelet packets using partitioning.

    Parameters:
    ----------
    G : np.ndarray
        Adjacency matrix of the graph.
    method : str, optional
        Laplacian method: 'L' (unnormalized) or 'Lrw' (random-walk normalized). Default is 'Lrw'.
    swapRegion : bool, optional
        Whether to allow region swapping to improve partition quality. Default is True.
    jmax : int, optional
        Maximum depth of the binary partition tree. If None, computed based on graph size.

    Returns:
    -------
    rs : np.ndarray
        Region boundaries per tree level.
    inds : np.ndarray
        Node indices corresponding to each region at each level.
    """
    N = len(G)
    if jmax == None:
        jmax = int(max(3 * np.floor(np.log2(N)), 4)) # jmax >= 4 is guaranteed.
    ind = np.arange(0, N)
    inds = np.zeros((N, jmax), dtype=int)
    inds[:, 0] = ind
    rs = np.zeros((N, jmax), dtype=int)
    rs[0, :] = 0
    rs[1, 0] = N
    
    j = 0
    regioncount = 0
    rs0 = 0
    while regioncount < N:
        regioncount = np.count_nonzero(rs[:, j]) # the number of regions at level j; I'd say the number of set of dual graph nodes
        print(regioncount)
        # print(j)
        if j == jmax-1:
            # tmp = np.zeros((N+1, 1), dtype=int); tmp[0] = 1
            # # rs = np.hstack((rs, np.vstack((np.ones(N+1, dtype=int), np.zeros(N+1, dtype=int)))))
            # rs = np.hstack((rs, tmp))
            # inds = np.hstack((inds, np.zeros((N, 1), dtype=int)))
            # jmax += 1
            break
        rr = 0
        for r in range(0, regioncount):
            rs1 = rs[r, j]
            rs2 = rs[r + 1, j]
            n = rs2 - rs1
            if n > 1:
                indrs = copy.deepcopy(ind[rs1:rs2])
                # partition the current region
                pm, _ = partition_fiedler(G[np.ix_(indrs, indrs)], method=method)
                # determine the number of poinInt in child region 1
                n1 = np.sum(pm > 0)
                if r > 1 and swapRegion:
                    if np.sum(G[np.ix_(ind[rs0:rs1 - 1], indrs[pm > 0])]) < np.sum(G[np.ix_(ind[rs0:rs1 - 1], indrs[pm < 0])]):
                        pm = -pm
                        n1 = n - n1
                # update the indexing
                ind[rs1:rs1 + n1] = indrs[pm > 0]
                ind[rs1 + n1:rs2] = indrs[pm < 0]
                rs[rr + 1, j + 1] = rs1 + n1
                rs[rr + 2, j + 1] = rs2
                rr += 2
                rs0 = rs1 + n1
            elif n == 1:
                rs[rr + 1, j + 1] = rs2
                rr += 1
                rs0 = rs1
        j += 1
        inds[:, j] = ind
    
    rs = rs[:, :j]
    inds = inds[:, :j]
    
    return rs, inds