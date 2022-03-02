import jax.numpy as np
from .normrow import normrow
from jax import jit

@jit
def normalizerow(X):
	"""
	NORMALIZEROW normalizes the l2-norm of each row in a np array 

	Input:
	  X: (n,m) numpy array
	Output:
	  X_normalized: (n,m) row normalized numpy array
	"""
	l2Norm = normrow(X)
	X_normalized = X / (l2Norm.reshape(X.shape[0],1))
	return X_normalized
