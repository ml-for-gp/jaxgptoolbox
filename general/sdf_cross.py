import numpy as onp
import jax.numpy as np

def sdf_cross(p, bx=0.35, by=0.12, r=0.):
    """
    output the signed distance value of a cross in 2D

    Inputs
    p: nx2 array of locations
    bx, by, r: parameters of the cross (please see the reference for more details)

    Outputs
    array of signed distance values at x

    Reference 
    https://iquilezles.org/www/articles/distfunctions2d/distfunctions2d.htm
    """
    p = onp.array(p - 0.5)
    p = onp.abs(p)
    p = onp.sort(p,1)[:,[1,0]]
    b = onp.array([bx, by])
    q = p - b
    k = onp.max(q, 1)
    w = q
    w[k<=0,0] = b[1] - p[k<=0,0]
    w[k<=0,1] = -k[k<=0]
    w = onp.maximum(w, 0.0)
    length_w = onp.sqrt(onp.sum(w*w, 1))
    out = onp.sign(k) * length_w + r
    return np.array(out)
