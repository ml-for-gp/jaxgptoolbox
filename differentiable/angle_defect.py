import jax.numpy as np
from .tip_angles import tip_angles
from ..general.boundary_vertices import boundary_vertices

def angle_defect(V,F,b=None,n=None):    
	'''
	ANGLE_DEFECT computes the angle defect (integrated Gauss curvature) at each
				 vertex

	Inputs:
	V: (|V|,3) numpy ndarray of vertex positions.
	F: (|F|,3) numpy ndarray of face indices, must be static.
	b: (|b|,) numpy ndarray of boundary vertex indices (they have 0 defect).
	   Will be computed if not provided, at the price of JITability.
	n: number of vertices in the mesh.
	   Will be computed if not provided, at the price of JITability.
	
	Outputs:
	k: (|N|,) numpy array of angle defect at each vertex
	'''
	#k = tip_angles(V,F)

	if b is None:
		bv = boundary_vertices(F)
	else:
		bv = b

	if n is None:
		nv,_ = V.shape
	else:
		nv = n

	k = angle_defect_intrinsic(F,tip_angles(V,F),bv,nv)

	#Pad return value if there are vertices in V not occuring in F
	# nv = V.shape[0]
	# nk = k.size
	# if nv>nk:
	# 	k = np.concatenate((k,np.zeros(nv-nk)))

	return k


def angle_defect_intrinsic(F,A,b=np.empty(0,dtype=int),n=None):    
	'''
	ANGLE_DEFECT_INTRINSIC computes the angle defect (integrated Gauss
						   curvature) at each vertex given intrinsic tip angles

	Inputs:
	F: (|F|,3) numpy ndarray of face indices
	A: (|F|,3) numpy ndarray of tip angles at each corner of each face (in [0,pi))
	b: (|b|,) numpy ndarray of boundary vertex indices (they have 0 defect)
	n: number of vertices in the mesh.
	   Will be computed if not provided, at the price of JITability.
	
	Outputs:
	k: (|N|,) numpy array of angle defect at each vertex
	'''

	if n is None:
		nv = F.max() + 1
	else:
		nv = n

	k = 2.*np.pi - np.bincount(np.ravel(F), weights=np.ravel(A), length=nv)
	# k = k.at[b].set(0)

	return k
