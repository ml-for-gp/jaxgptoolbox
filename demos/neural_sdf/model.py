import sys
sys.path.append('../../../')
import jaxgptoolbox as jgp

import jax
import jax.numpy as np
from jax import jit, value_and_grad
from jax.experimental import optimizers

import numpy as onp
import numpy.random as random
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tqdm
import pickle

class mlp:
  def __init__(self, hyperParams):
    self.hyperParams = hyperParams
  
  def initialize_weights(self):
    """
    Initialize the parameters of the Lipschitz mlp

    Inputs
    hyperParams: hyper parameter dictionary

    Outputs
    params_net: parameters of the network (weight, bias, initial lipschitz bound)
    """
    def init_W(size_out, size_in): 
        W = onp.random.randn(size_out, size_in) * onp.sqrt(2 / size_in)
        return np.array(W)
    sizes = self.hyperParams["h_mlp"]
    sizes.insert(0, self.hyperParams["dim_in"] + self.hyperParams["dim_t"])
    sizes.append(self.hyperParams["dim_out"])
    params_net = []    
    for ii in range(len(sizes) - 1):
        W = init_W(sizes[ii+1], sizes[ii])
        b = np.zeros(sizes[ii+1])
        params_net.append([W, b])
    return params_net

  def forward_single(self, params_net, t, x):
    """
    Forward pass of a lipschitz MLP
    
    Inputs
    params_net: parameters of the network
    t: the input feature of the shape
    x: a query location in the space

    Outputs
    out: implicit function value at x
    """
    # concatenate coordinate and latent code
    x = np.append(x, t)

    # forward pass
    for ii in range(len(params_net) - 1):
        W, b = params_net[ii]
        x = jax.nn.relu(np.dot(W, x) + b)

    # final layer
    W, b = params_net[-1]
    out = np.dot(W, x) + b
    return out[0]
  forward = jax.vmap(forward_single, in_axes=(None, None, None, 0), out_axes=0)