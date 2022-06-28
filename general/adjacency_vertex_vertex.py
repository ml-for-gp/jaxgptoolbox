import numpy as np
import scipy
from .edges import edges

def adjacency_vertex_vertex(F):
    """
    This function computes vertex to vertex adjacency matrix

    inputs:
    F: |F|x3 list of face indices

    outputs:
    A: |V|x|V| scipy sparse matrix of vertex adjacency matrix 
    """
    E = edges(F)
    nV = F.max() + 1

    row = np.concatenate((E[:,0],E[:,1]))
    col = np.concatenate((E[:,1],E[:,0]))
    val = np.ones(len(row), dtype=np.int)
    A = scipy.sparse.coo_matrix((val,(row, col)), shape=(nV,nV)).tocsr()
    return A