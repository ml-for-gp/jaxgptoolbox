import jax.numpy as np
from jax import jit

@jit
def vertex_areas(V,F):
    """
    computes area per vertex 

    Input:
        V (|V|,3) numpy array of vertex positions
        F (|F|,3) numpy array of face indices
    Output:
        VA (|V|,) numpy array of vertex areas
    """
    l1 = np.sqrt(np.sum((V[F[:,1],:]-V[F[:,2],:])**2,1))
    l2 = np.sqrt(np.sum((V[F[:,2],:]-V[F[:,0],:])**2,1))
    l3 = np.sqrt(np.sum((V[F[:,0],:]-V[F[:,1],:])**2,1))

    cos1 = (l3**2+l2**2-l1**2) / (2*l2*l3)
    cos2 = (l1**2+l3**2-l2**2) / (2*l1*l3)
    cos3 = (l1**2+l2**2-l3**2) / (2*l1*l2)

    cosMat = np.concatenate( (cos1[:,None], cos2[:,None], cos3[:,None]), axis =1)
    lMat = np.concatenate( (l1[:,None], l2[:,None], l3[:,None]), axis =1)
    barycentric = cosMat * lMat
    normalized_barycentric = barycentric / np.sum(barycentric,1)[:,None]
    areas = 0.25 * np.sqrt( (l1+l2-l3)*(l1-l2+l3)*(-l1+l2+l3)*(l1+l2+l3) )
    partArea = normalized_barycentric * areas[:,None]

    quad0 = (partArea[:,1]+partArea[:,2]) * 0.5
    quad1 = (partArea[:,0]+partArea[:,2]) * 0.5
    quad2 = (partArea[:,0]+partArea[:,1]) * 0.5

    idx = np.where(cos1<0, 0, 1)
    quad0 = quad0.at[idx].set(areas[idx] * 0.5)
    quad1 = quad1.at[idx].set(areas[idx] * 0.25)
    quad2 = quad2.at[idx].set(areas[idx] * 0.25)

    idx = np.where(cos2<0, 0, 1)
    quad0 = quad0.at[idx].set(areas[idx] * 0.25)
    quad1 = quad1.at[idx].set(areas[idx] * 0.5)
    quad2 = quad2.at[idx].set(areas[idx] * 0.25)

    idx = np.where(cos3<0, 0, 1)
    quad0 = quad0.at[idx].set(areas[idx] * 0.25)
    quad1 = quad1.at[idx].set(areas[idx] * 0.25)
    quad2 = quad2.at[idx].set(areas[idx] * 0.5)
    quads = np.concatenate( (quad0[:,None], quad1[:,None], quad2[:,None]), axis =1).flatten()

    VA = np.zeros((V.shape[0],))
    VA = VA.at[F.flatten()].add(quads)
    return VA