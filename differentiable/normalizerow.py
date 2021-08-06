import jax.numpy as np

def normalizerow(X):
	"""
	NORMALIZEROW normalizes the l2-norm of each row in a np array 

	Input:
	  X: n-by-m np array
	Output:
	  X_normalized: n-by-m row normalized np array
	"""
	l2Norm = np.sqrt((X * X).sum(axis=1))
	X_normalized = X / (l2Norm.reshape(X.shape[0],1))
	return X_normalized