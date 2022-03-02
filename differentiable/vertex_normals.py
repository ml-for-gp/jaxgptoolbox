import jax.numpy as np
from jax import jit
from .normalizerow import normalizerow

@jit
def vertex_normals(V,F):
    """
    Computes face area weighted vertex normals

    Input:
    V |V|x3 numpy array of vertex positions
    F |F|x3 numpy array of face indices

    Output:
    N |V|x3 array of normalized vertex normal (weighted by face areas)
    """
    vec1 = V[F[:,1],:] - V[F[:,0],:]
    vec2 = V[F[:,2],:] - V[F[:,0],:]
    FN_unnormalized = np.cross(vec1, vec2) / 2

    VN = np.zeros((V.shape[0],3), dtype=np.float32)
    VN = VN.at[F[:,0]].add(FN_unnormalized)
    VN = VN.at[F[:,1]].add(FN_unnormalized)
    VN = VN.at[F[:,2]].add(FN_unnormalized)
    return normalizerow(VN)
