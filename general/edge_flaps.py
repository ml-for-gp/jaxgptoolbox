import numpy as np
from . edges_with_mapping import edges_with_mapping

def edge_flaps(F):
  '''
  EDGEFLAPS compute flap edge indices for each edge 

  Input:
    F (|F|,3) numpy array of face indices
  Output:
    E (|E|,2) numpy array of edge indices
    flapEdges (|E|, 4 or 2) numpy array of edge indices
  '''
  # Notes:
  # Each flapEdges[e,:] = [a,b,c,d] edges indices 
  #    / \
  #   b   a
  #  /     \
  #  - e - -
  #  \     /
  #   c   d
  #    \ /
  E, F2E = edges_with_mapping(F)
  flapEdges = [[] for i in range(E.shape[0])]
  for f in range(F.shape[0]):
    e0 = F2E[f,0]
    e1 = F2E[f,1]
    e2 = F2E[f,2]
    flapEdges[e0].extend([e1,e2])
    flapEdges[e1].extend([e2,e0])
    flapEdges[e2].extend([e0,e1])
  return E, np.array(flapEdges)