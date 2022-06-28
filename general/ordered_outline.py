import numpy as np
from .outline import outline

def ordered_outline(ForO):
    """
    this function computes an ordered boundary curves of a mesh

    Inputs:
    F: |F|x3 face index list 
    or
    O: |E|x2 unordered outline

    Outputs
    L: list of list of order boundary indices such that L[0] is the list of vertices of the 0th boundary curve
    """
    if ForO.shape[1] == 3: # input is a face list
        F = np.array(ForO)
        O = outline(F)
        O = np.array(O)
    elif ForO.shape[1] == 2: # input is a (unordered) boundary curve list
        O = np.array(ForO)

    # index map for vertices such that IMV[old_vIdx] = new_vIdx
    uV = np.unique(O)
    nV = O.max() + 1
    IMV = np.zeros(nV, dtype = np.int64)
    IMV[uV] = np.arange(len(uV))

    # inverse index map such that invIMV[new_vIdx] = old_vIdx
    invIMV = uV

    # index map to O[:,0] such that old_vIdx = O[IMO[old_vIdx],0]
    IMO = np.zeros(nV, dtype = np.int64)
    IMO[O[:,0]] = np.arange(len(uV))

    L = [] # loop for multiple boundary loops
    visited = np.full((len(uV),),False) # whether visited (stored in new vIdx)
    while not np.all(visited):
        # get a vertex for a Loop
        vnew = np.where(visited == False)[0][0]
        v = invIMV[vnew]
        next_v = get_next_v(v,O,IMO)

        start_v = v # track starting vertex
        B = [] # to store each boundary loop
        B.append(start_v) # add the starting vertex

        # update visited list
        visited[IMV[start_v]] = True
        

        # find all the vertices of this loop
        while next_v != start_v:
            B.append(next_v) # add the next vertex
            visited[IMV[next_v]] = True
            v = next_v
            next_v = get_next_v(v,O,IMO)
        L.append(B)
    return L 

def get_next_v(v,O,IMO):
    return O[IMO[v],1]