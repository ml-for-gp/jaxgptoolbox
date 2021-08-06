import jax.numpy as np
from .tip_angles import tip_angles
from ..general.boundary_vertices import boundary_vertices

def angle_defect(V,F):    
	'''
	ANGLE_DEFECT computes the angle defect (integrated Gauss curvature) at each
				 vertex

	Inputs:
	V: (|V|,3) numpy ndarray of vertex positions
	F: (|F|,3) numpy ndarray of face indices
	
	Outputs:
	k: (|N|,) numpy array of angle defect at each vertex
	'''

	k = angle_defect_intrinsic(F,tip_angles(V,F),boundary_vertices(F))

	#Pad return value if there are vertices in V not occuring in F
	nv = V.shape[0]
	nk = k.size
	if nv>nk:
		k = np.concatenate((k,np.zeros(nv-nk)))

	return k


def angle_defect_intrinsic(F,A,b=np.empty(0,dtype=int)):    
	'''
	ANGLE_DEFECT_INTRINSIC computes the angle defect (integrated Gauss
						   curvature) at each vertex given intrinsic tip angles

	Inputs:
	F: (|F|,3) numpy ndarray of face indices
	A: (|F|,3) numpy ndarray of tip angles at each corner of each face (in [0,pi))
	b: (|b|,) numpy ndarray of boundary vertex indices (they have 0 defect)
	
	Outputs:
	k: (|N|,) numpy array of angle defect at each vertex
	'''

	k = 2.*np.pi - np.bincount(np.ravel(F), weights=np.ravel(A))
	k = k.at[b].set(0)
	return k
