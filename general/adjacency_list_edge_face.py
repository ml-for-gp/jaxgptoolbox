import numpy as np
import jax
from . adjacency_edge_face import adjacency_edge_face

def adjacency_list_edge_face(F):
  '''
  ADJACENCY_LIST_EDGE_FACE computes edge-face adjacency list

  Input:
    F (|F|,3) numpy array of face indices
  Output:
    adjList (|E|,2) numpy array of adjacency list between edges and faces
    E (|E|,2) numpy array of edge indices
  '''

  E2F, E = adjacency_edge_face(F)
  adjList = np.zeros((E.shape[0], 2), dtype=np.int) # assume manifold
  E2F_bool = E2F.astype('bool')

  idx = np.arange(F.shape[0])

  for e in range(E.shape[0]):
    E2F_bool_row = np.squeeze(np.asarray(E2F_bool[e,:].todense()))
    adjList[e,:] = idx[E2F_bool_row]

  return jax.numpy.asarray(adjList), E