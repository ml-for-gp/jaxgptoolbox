import jax.numpy as np
import numpy as onp
import igl

def read_mesh(path):    
  '''
  just a jax wrapper for igl.read_triangle_mesh
  '''
  V, F = igl.read_triangle_mesh(path)
  return np.array(V), np.array(F)