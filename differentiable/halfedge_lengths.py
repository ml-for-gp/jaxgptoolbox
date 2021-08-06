import jax.numpy as np
from .normalizerow import normalizerow

def halfedge_lengths(V, F):    
	'''
	HALFEDGE_LENGTHS computes the lengths of all halfedges in the mesh

	Inputs:
	V: (|V|,3) numpy ndarray of vertex positions
	F: (|F|,3) numpy ndarray of face indices
	
	Outputs:
	l: (|F|,3) numpy ndarray of halfedge lenghts.
			   Our halfedge convention identifies each halfedge by the index
			   of the face and the opposite vertex within the face:
			   (face, opposite vertex)
	'''

	he0 = V[F[:,2],:] - V[F[:,1],:]
	he1 = V[F[:,0],:] - V[F[:,2],:]
	he2 = V[F[:,1],:] - V[F[:,0],:]

	l = np.concatenate((normalizerow(he0),normalizerow(he1),normalizerow(he2)),
		axis=1)

	return l
