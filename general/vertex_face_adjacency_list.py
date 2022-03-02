import numpy as onp

def vertex_face_adjacency_list(F):
    """
    build a vertex-face adjacency list such that V2F[vertex_index] = [adjacent_face_indices]

    Inputs
    F: |F|x3 array of face indices

    Outputs
    V2F: list of lists with so that V2F[v] = [fi, fj, ...]
    """
    F = onp.array(F)
    nV = F.max() + 1
    V2F = [[] for _ in range(nV)]
    for f in range(F.shape[0]):
        for s in range(F.shape[1]):
            V2F[F[f,s]].append(f)
    return V2F