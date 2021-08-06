import numpy as np
import jax
from .outline import outline

def boundary_vertices(F):
	'''
	OUTLINE compute the unordered outline edges of a triangle mesh

	Input:
	  F (|F|,3) numpy array of face indices
	Output:
	  b (|bE|,) numpy array of boundary vertex indices
	'''

	return np.unique(outline(F))
