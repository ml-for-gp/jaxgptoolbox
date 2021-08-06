import numpy as np

def edges_with_mapping(F):
  '''
	EDGES compute edges from face information

	Input:
	  F (|F|,3) numpy array of face indices
	Output:
	  uE (|E|,2) numpy array of edge indices
    F2E (|F|,3) numpy array mapping from halfedges to unique edges.
                Our halfedge convention identifies each halfedge by the index
                of the face and the opposite vertex within the face:
                (face, opposite vertex)
	'''
  F12 = F[:, np.array([1,2])]
  F20 = F[:, np.array([2,0])]
  F01 = F[:, np.array([0,1])]
  EAll = np.concatenate( (F12, F20, F01), axis = 0)

  EAll_sortrow = np.sort(EAll, axis = 1)
  uE, F2E_vec = np.unique(EAll_sortrow, return_inverse=True, axis=0)
  F2E = F2E_vec.reshape(F.shape[1], F.shape[0]).T
  return uE, F2E