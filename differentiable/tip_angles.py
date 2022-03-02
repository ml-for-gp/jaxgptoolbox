import jax.numpy as np
from jax import jit
from .halfedge_lengths import halfedge_lengths_squared

@jit
def tip_angles(V, F):    
	'''
	TIP_ANGLES computes the tip angles of a mesh (the angle at every corner)

	Inputs:
	V: (|V|,3) numpy ndarray of vertex positions
	F: (|F|,3) numpy ndarray of face indices
	
	Outputs:
	A: (|F|,3) numpy ndarray of tip angles at each corner of each face (in [0,pi))
	'''

	return tip_angles_intrinsic(halfedge_lengths_squared(V,F))


def tip_angles_intrinsic(l_sq):    
	'''
	TIP_ANGLES_INTRINSIC computes the tip angles of a mesh (the angle at every
						 corner) given squared halfedge lengths

	Inputs:
	l_sq: (|F|,3) numpy ndarray of squared halfedge lenghts.
			   Our halfedge convention identifies each halfedge by the index
			   of the face and the opposite vertex within the face:
			   (face, opposite vertex)
	
	Outputs:
	A: (|F|,3) numpy ndarray of tip angles at each corner of each face (in [0,pi))
	'''
	
	#Use the cosine rule
	a_sq = np.expand_dims(l_sq[:,0], axis=1)
	b_sq = np.expand_dims(l_sq[:,1], axis=1)
	c_sq = np.expand_dims(l_sq[:,2], axis=1)
	a = np.sqrt(a_sq)
	b = np.sqrt(b_sq)
	c = np.sqrt(c_sq)

	cos_angles = np.concatenate((
			(b_sq + c_sq - a_sq) / (2.*b*c),
			(a_sq + c_sq - b_sq) / (2.*a*c),
			(a_sq + b_sq - c_sq) / (2.*a*b)
		), axis=1).clip(-1,1)

	return np.arccos(cos_angles)
