import numpy as np
import scipy
import jax
from . edges_with_mapping import edges_with_mapping

def adjacency_edge_face(F):
  '''
  ADJACENCY_EDGE_FACE computes edge-face adjacency matrix

  Input:
    F (|F|,3) numpy array of face indices
  Output:
    E2F (|E|, |F|) scipy sparse matrix of adjacency information between edges and faces
    E (|E|,2) numpy array of edge indices
  '''
  E, F2E = edges_with_mapping(F)
  IC = F2E.T.reshape(F.shape[1]*F.shape[0])

  row = IC
  col = np.tile(np.arange(F.shape[0]), 3)
  val = np.ones(len(IC), dtype=np.int)
  E2F = scipy.sparse.coo_matrix((val,(row, col)), shape=(E.shape[0], F.shape[0])).tocsr()

  return E2F, jax.numpy.asarray(E)
