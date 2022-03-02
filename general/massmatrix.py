import numpy as np
import scipy
import scipy.sparse 

def massmatrix(V, F):
    """
    construct voronoi area mass matrix of vertex areas

    Input:
    V (|V|,3) array of vertex positions
    F (|F|,3) array of face indices

    Output:
    M (|V|,|V|) scipy sparse diagonal matrix of vertex area
    """
    # in case input is not numpy array
    V = np.array(V)
    F = np.array(F)

    l1 = np.sqrt(np.sum((V[F[:,1],:]-V[F[:,2],:])**2,1))
    l2 = np.sqrt(np.sum((V[F[:,2],:]-V[F[:,0],:])**2,1))
    l3 = np.sqrt(np.sum((V[F[:,0],:]-V[F[:,1],:])**2,1))
    lMat = np.concatenate( (l1[:,None], l2[:,None], l3[:,None]), axis =1)

    cos1 = (l3**2+l2**2-l1**2) / (2*l2*l3)
    cos2 = (l1**2+l3**2-l2**2) / (2*l1*l3)
    cos3 = (l1**2+l2**2-l3**2) / (2*l1*l2)
    cosMat = np.concatenate( (cos1[:,None], cos2[:,None], cos3[:,None]), axis =1)

    barycentric = cosMat * lMat
    normalized_barycentric = barycentric / np.sum(barycentric,1)[:,None]
    areas = 0.25 * np.sqrt( (l1+l2-l3)*(l1-l2+l3)*(-l1+l2+l3)*(l1+l2+l3) )
    partArea = normalized_barycentric * areas[:,None]
    quad1 = (partArea[:,1]+partArea[:,2]) * 0.5
    quad2 = (partArea[:,0]+partArea[:,2]) * 0.5
    quad3 = (partArea[:,0]+partArea[:,1]) * 0.5
    quads = np.concatenate( (quad1[:,None], quad2[:,None], quad3[:,None]), axis =1)

    boolM = cosMat[:,0]<0
    quads[boolM,0] = areas[boolM]*0.5
    quads[boolM,1] = areas[boolM]*0.25
    quads[boolM,2] = areas[boolM]*0.25

    boolM = cosMat[:,1]<0
    quads[boolM,0] = areas[boolM]*0.25
    quads[boolM,1] = areas[boolM]*0.5
    quads[boolM,2] = areas[boolM]*0.25

    boolM = cosMat[:,2]<0
    quads[boolM,0] = areas[boolM]*0.25
    quads[boolM,1] = areas[boolM]*0.25
    quads[boolM,2] = areas[boolM]*0.5

    rIdx = F.flatten()
    cIdx = F.flatten()
    val = quads.flatten()
    nV = V.shape[0]
    M = scipy.sparse.csr_matrix( (val,(rIdx,cIdx)), shape=(nV,nV), dtype=np.float64 )
    return M