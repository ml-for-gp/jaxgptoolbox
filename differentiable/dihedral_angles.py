import jax.numpy as np
from .face_normals import face_normals
from .dotrow import dotrow
from ..general.adjacency_list_edge_face import adjacency_list_edge_face

def dihedral_angles(V, F):    
	'''
	DIHEDRAL_ANGLES computes the dihedral angles of a mesh

	Inputs:
	V: (|V|,3) numpy ndarray of vertex positions
	F: (|F|,3) numpy ndarray of face indices
	
	Outputs:
	dihedral_angles: (|E|,) numpy array of dihedral angles (0 ~ pi)
	E: (|E|,2) numpy array of edge indices
	'''
	# TODO: double check whether this is differentiable

	FN = face_normals(V,F)
	E2F, E = adjacency_list_edge_face(F)
	return dihedral_angles_from_normals(FN,E2F), E


def dihedral_angles_from_normals(N, E2F):    
	'''
	DIHEDRAL_ANGLES_FROM_NORMALS computes the dihedral angles of a mesh, given
								 precomputed normals and edge-to-face map

	Inputs:
	N: (|F|,3) numpy ndarray of unit face normals
	E2F: (|E|,|F|) scipy sparse matrix scipy sparse matrix of adjacency
		 information between edges and faces, for example as produced by
		 adjacency_list_edge_face
	
	Outputs:
	dihedral_angles: (|E|,) numpy array of dihedral angles at each edge 
	'''

	dotN = dotrow(N[E2F[:,0],:], N[E2F[:,1],:]).clip(-1,1)
	return np.pi - np.arccos(dotN)
