import jax.numpy as np
import numpy as onp
import igl

def signed_distance(P, V, F):    
  '''
  SIGNED_DISTANCE computes signed distance from given points to a mesh

  Inputs:
  P: (|P|,3) numpy ndarray of point positions
  V: (|V|,3) numpy ndarray of vertex positions
  F: (|F|,3) numpy ndarray of face indices

  Outputs:
  S: (|P|,) numpy array of signed distance
  pF:(|P|,) numpy array of projected face indices
  pV:(|P|,3) numpy array of projected point locations

  Notes:
  It can be differentiable, but this faster version based on C++ is not
  '''
  P_np = onp.array(P)
  V_np = onp.array(V)
  F_np = onp.array(F)
  S,pF,pV = igl.signed_distance(P_np, V_np, F_np)

  return np.array(S), np.array(pF), np.array(pV)