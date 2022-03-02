import jax.numpy as np
from .dotrow import dotrow
from jax import jit

@jit
def normrow(X):
	"""
	NORMROW computes the l2-norm of each row in a np array 

	Input:
	  X: (n,m) numpy array
	Output:
	  nX: (n,) numpy array of l2 norm of each row in X
	"""

	return np.sqrt(dotrow(X,X))