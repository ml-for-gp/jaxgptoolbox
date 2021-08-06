import jax.numpy as np
from .dotrow import dotrow

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

	return np.sqrt(halfedge_lengths_squared(V,F))


def halfedge_lengths_squared(V, F):    
	'''
	HALFEDGE_LENGTHS_SQUARED computes the lengths of all halfedges in the mesh,
							 squared (this is often preferable to just the
							 lengths, since it's easier to differentiate
							 through)

	Inputs:
	V: (|V|,3) numpy ndarray of vertex positions
	F: (|F|,3) numpy ndarray of face indices
	
	Outputs:
	l_sq: (|F|,3) numpy ndarray of squared halfedge lenghts.
			   Our halfedge convention identifies each halfedge by the index
			   of the face and the opposite vertex within the face:
			   (face, opposite vertex)
	'''

	he0 = V[F[:,2],:] - V[F[:,1],:]
	he1 = V[F[:,0],:] - V[F[:,2],:]
	he2 = V[F[:,1],:] - V[F[:,0],:]

	lhe0 = np.expand_dims(dotrow(he0,he0), axis=1)
	lhe1 = np.expand_dims(dotrow(he1,he1), axis=1)
	lhe2 = np.expand_dims(dotrow(he2,he2), axis=1)

	l_sq = np.concatenate((lhe0, lhe1, lhe2), axis=1)

	return l_sq
