import numpy as onp
from scipy.sparse import coo_matrix

def adjacency_list_face_face(F):
    """
    build a face-face adjacency list such that F2F[face_index] = [adjacent_face_indices]. Note that neighboring faces are determined by whether two faces share an edge.

    Inputs
    F: |F|x3 array of face indices

    Outputs
    F2F: list of lists with so that F2F[f] = [fi, fj, ...]
    """

    F = onp.array(F)

    # build adjacency matrix
    E_all = onp.vstack((F[:,[1,2]], F[:,[2,0]], F[:,[0,1]]))
    sort_E_all = onp.sort(E_all,-1)

    # extract unique edges with inverse indices s.t. E[inv_idx,:] = sort_E_all
    E, inv_idx = onp.unique(sort_E_all, axis=0, return_inverse=True)

    nF = F.shape[0]
    nE = E.shape[0]

    row = inv_idx
    col = onp.repeat(onp.arange(nF),F.shape[1])
    data = onp.ones_like(inv_idx)
    uE2F = coo_matrix((data, (row, col)), shape=(nE, nF))
    F2F_mat =  uE2F.T  @ uE2F
    F2F_mat.setdiag(0)

    F2F = [[] for _ in range(nF)]
    for f in range(nF):
        _, col = F2F_mat[f,:].nonzero()
        F2F[f] = col.tolist()

    return F2F