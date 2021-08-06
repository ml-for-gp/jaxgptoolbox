import jax.numpy as np
from . face_normals import face_normals
from ..general.adjacency_list_edge_face import adjacency_list_edge_face

def dihedral_angles(V,F):
  '''
  DIHEDRAL_ANGLES computes dihedral angles of a triangle mesh

  Inputs:
    V (|V|,3) numpy ndarray of vertex positions
    F (|F|,3) numpy ndarray of face indices
  Outputs:
    dihedral_angles: (|E|,) numpy array of dihedral angles (0 ~ pi)
    E: (|E|,2) numpy array of edge indices
  '''
  # TODO: double check whether this is differentiable

  FN = face_normals(V,F)
  E2F, E = adjacency_list_edge_face(F)
  dotN = np.sum(FN[E2F[:,0],:] * FN[E2F[:,1],:], axis = 1).clip(-1, 1) # clip to avoid nan
  return np.pi - np.arccos(dotN), E