import numpy as np
import jax

def outline(F):
	'''
	OUTLINE compute the unordered outline edges of a triangle mesh

	Input:
	  F (|F|,3) numpy array of face indices
	Output:
	  O (|bE|,2) numpy array of boundary vertex indices, one edge per row
	'''

	# All halfedges
	he = np.stack((np.ravel(F[:,[1,2,0]]),np.ravel(F[:,[2,0,1]])), axis=1)

	# Sort hes to be able to find duplicates later
	# inds = np.argsort(he, axis=1)
	# he_sorted = np.sort(he, inds, axis=1)
	he_sorted = np.sort(he, axis=1)

	# Extract unique rows
	_,unique_indices,unique_counts = np.unique(he_sorted, axis=0, return_index=True, return_counts=True)

	# All the indices with only one unique count are the original boundary edges
	# in he
	O = he[unique_indices[unique_counts==1],:]

	return O
