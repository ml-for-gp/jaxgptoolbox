import numpy as onp
import jax.numpy as np

def sdf_star(x, r = 0.22):
    """
    output the signed distance value of a star in 2D

    Inputs
    x: nx2 array of locations
    r: size of the star

    Outputs
    array of signed distance values at x

    Reference:
    https://iquilezles.org/www/articles/distfunctions2d/distfunctions2d.htm
    """
    x = onp.array(x)
    kxy = onp.array([-0.5,0.86602540378])
    kyx = onp.array([0.86602540378,-0.5])
    kz = 0.57735026919
    kw = 1.73205080757

    x = onp.abs(x - 0.5)
    x -= 2.0 * onp.minimum(x.dot(kxy), 0.0)[:,None] * kxy[None,:]
    x -= 2.0 * onp.minimum(x.dot(kyx), 0.0)[:,None] * kyx[None,:]
    x[:,0] -= onp.clip(x[:,0],r*kz,r*kw)
    x[:,1] -= r
    length_x = onp.sqrt(onp.sum(x*x, 1))
    return np.array(length_x*onp.sign(x[:,1]))