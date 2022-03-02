import jax.numpy as np
from jax import jit

@jit
def dotrow(X,Y):    
	'''
	DOTROW computes the row-wise dot product of the rows of two matrices

	Inputs:
	X: (n,m) numpy ndarray
	Y: (n,m) numpy ndarray
	
	Outputs:
	d: (n,) numpy array of rowwise dot product of X and Y
	'''

	return np.sum(X * Y, axis = 1)

