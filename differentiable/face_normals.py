import jax.numpy as np
from .normalizerow import normalizerow

def face_normals(V, F):    
	'''
	FACENORMALS computes unit face normal of a triangle mesh

	Inputs:
	V: n-by-3 numpy ndarray of vertex positions
	F: m-by-3 numpy ndarray of face indices
	
	Outputs:
	FN_normalized: m-by-3 numpy ndarray of unit face normal
	'''
	vec1 = V[F[:,1],:] - V[F[:,0],:]
	vec2 = V[F[:,2],:] - V[F[:,0],:]
	FN = np.cross(vec1, vec2) / 2
	l2Norm = np.sqrt((FN * FN).sum(axis=1))
	# FN_normalized = FN / (l2Norm.reshape(FN.shape[0],1))
	FN_normalized = normalizerow(FN)
	return FN_normalized