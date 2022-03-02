import jax.numpy as np
from jax import jit

@jit
def cotangent_weights(V,F):
    """
    computes cotangent weight for each half edge

    Input:
        V (|V|,3) array of vertex positions
        F (|F|,3) array of face indices
    Output:
        C (|F|,3) array of cotengent weights
    """
    i0 = F[:,0]
    i1 = F[:,1]
    i2 = F[:,2]
    l0 = np.sqrt(np.sum(np.power(V[i1,:] - V[i2,:],2),1))
    l1 = np.sqrt(np.sum(np.power(V[i2,:] - V[i0,:],2),1))
    l2 = np.sqrt(np.sum(np.power(V[i0,:] - V[i1,:],2),1))

    # Heron's formula for area
    s = (l0 + l1 + l2) / 2.
    area = np.sqrt(s * (s-l0) * (s-l1) * (s-l2))

    # cotangent weights
    c0 = (l1*l1 + l2*l2 - l0*l0) / area / 8
    c1 = (l0*l0 + l2*l2 - l1*l1) / area / 8
    c2 = (l0*l0 + l1*l1 - l2*l2) / area / 8
    C = np.concatenate((c0[:,None], c1[:,None], c2[:,None]), axis = 1)
    return C