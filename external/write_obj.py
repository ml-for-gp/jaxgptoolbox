import jax.numpy as np
import numpy as onp
import igl

def write_obj(path, V, F):    
  '''
  just a jax wrapper for igl.write_obj
  '''
  igl.write_obj(path,onp.array(V),onp.array(F))