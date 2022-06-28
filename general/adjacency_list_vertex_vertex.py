import numpy as np
from .adjacency_vertex_vertex import adjacency_vertex_vertex

def adjacency_list_vertex_vertex(F):
    """
    This function computes vertex to vertex adjacency matrix

    inputs:
    F: |F|x3 list of face indices

    outputs:
    AList: |V| list of adjacent vertex indices
    """
    A = adjacency_vertex_vertex(F)
    AList = []

    for ii in range(A.shape[0]):
        Aii_nonzero = A[ii,:].nonzero()[1]
        AList.append(Aii_nonzero)

    return AList