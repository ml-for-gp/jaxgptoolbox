import jax.numpy as np 
from jax import jit

@jit
def normalize_unit_box(V, margin = 0.0):
  """
	NORMALIZE_UNIT_BOX normalize a set of points to a unit bounding box with a user-specified margin 

	Input:
	  V: (n,3) numpy array of point locations
    margin: a constant of user specified margin
	Output:
	  V: (n,3) numpy array of point locations bounded by margin ~ 1-margin
	"""

  V = V - V.min(0) 
  V = V / V.max()
  V = V - 0.5
  V = V * (1.0 - margin*2)
  V = V + 0.5
  return V