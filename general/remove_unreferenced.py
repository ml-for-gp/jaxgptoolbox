import numpy as np

def remove_unreferenced(V,F):
    """
    remove invalid faces, [-1,-1,-1], and unreferenced vertices from the intrinsic mesh

    Inputs
    V: |V|x3 array of vertex locations
    F: |F|x3 array of face list

    Outputs
    V,F: new mesh
    IMV: index map for vertices such that IMV[old_vIdx] = new_vIdx
    """
    V = np.array(V)
    F = np.array(F)

    # removed unreferenced vertices/faces
    nV = V.shape[0]

    # get a list of unique vertex indices from face list
    uV = np.unique(F)

    # index map for vertices such that IMV[old_vIdx] = new_vIdx
    IMV = np.zeros(nV, dtype = np.int64)
    IMV[uV] = np.arange(len(uV))

    # return the new mesh
    V = V[uV]
    F = IMV[F]
    return V,F,IMV