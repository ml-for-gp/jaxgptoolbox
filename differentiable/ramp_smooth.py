import jax.numpy as np

def ramp_smooth(d, d0 = 1.0):
    """
    RAMP function so that: 
    - 1, when d/d0 >= 1 
    - -1 when d/d0 <= 1
    - smooth decay when -1 < d/d0 < 1
    (following the notation of "Curl-Noise for Procedural Fluid Flow" Bridson et al 2007)

    Input:
        d      distance of a query point to its closest point on boundary
        d0     scale of the decay
    Output:
        scale  ramp scaling factor
    """
    r = d / d0
    val = 15./8.*r - 10./8.*(r**3) + 3./8.*(r**5)
    return np.minimum(np.abs(val), 1.0) * np.sign(val)