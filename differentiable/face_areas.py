import jax.numpy as np
from jax import jit

@jit
def face_areas(V, F):
    """
    FACEAREAS computes area per face 

    Input:
        V (|V|,3) numpy array of vertex positions
        F (|F|,3) numpy array of face indices
    Output:
        FA (|F|,) numpy array of face areas
    """
    vec1 = V[F[:,1],:] - V[F[:,0],:]
    vec2 = V[F[:,2],:] - V[F[:,0],:]
    FN = np.cross(vec1, vec2) / 2
    FA = np.sqrt(np.sum(FN**2,1))
    return FA