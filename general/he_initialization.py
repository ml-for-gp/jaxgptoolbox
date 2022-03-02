import numpy as onp
import jax.numpy as np
def he_initialization(n_out, n_in):
    """
    initialization based on He et al 2015. This is often recommended for ReLU like networks

    Inputs
    n_out: dimension of the output of a linear layer
    n_in:  dimension of the input of a linear layer

    Outputs
    W: jax array of weight matrix
    b: jax array of bias matrix
    """
    W = np.array(onp.random.randn(n_out, n_in) * onp.sqrt(2 / n_in))
    b = np.zeros(n_out)
    return W, b