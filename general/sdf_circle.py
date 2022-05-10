import numpy as onp
import jax.numpy as np

def sdf_circle(x, r = 0.282, center = np.array([0.5,0.5])):
    """
    output the SDF value of a circle in 2D

    Inputs
    x: nx2 array of locations
    r: radius of the circle
    center: center point of the circle

    Outputs
    array of signed distance values at x
    """
    dx = x - center
    return np.sqrt(np.sum((dx)**2, axis = 1)) - r