import sys
sys.path.append('../../../')
import jaxgptoolbox as jgp

import jax
import jax.numpy as np
from jax import jit, value_and_grad
from jax.example_libraries import optimizers


import numpy as onp
import numpy.random as random
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import tqdm
import pickle

class lipmlp:
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
        c = np.max(np.sum(np.abs(W), axis=1))
        params_net.append([W, b, c])
    return params_net

  def weight_normalization(self, W, softplus_c):
    """
    Lipschitz weight normalization based on the L-infinity norm
    """
    absrowsum = np.sum(np.abs(W), axis=1)
    scale = np.minimum(1.0, softplus_c/absrowsum)
    return W * scale[:,None]

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
        W, b, c = params_net[ii]
        W = self.weight_normalization(W, jax.nn.softplus(c))
        x = jax.nn.relu(np.dot(W, x) + b)

    # final layer
    W, b, c = params_net[-1]
    W = self.weight_normalization(W, jax.nn.softplus(c)) 
    out = np.dot(W, x) + b
    return out[0]
  forward = jax.vmap(forward_single, in_axes=(None, None, None, 0), out_axes=0)

  def get_lipschitz_loss(self, params_net):
    """
    This function computes the Lipschitz regularization
    """
    loss_lip = 1.0
    for ii in range(len(params_net)):
      W, b, c = params_net[ii]
      loss_lip = loss_lip * jax.nn.softplus(c)
    return loss_lip

  def normalize_params(self, params_net):
    """
    (Optional) After training, this function will clip network [W, b] based on learned lipschitz constants. Thus, one can use normal MLP forward pass during test time, which is a little bit faster.
    """
    params_final = []    
    for ii in range(len(params_net)):
      W, b, c = params_net[ii]
      W = self.weight_normalization(W, jax.nn.softplus(c))
      params_final.append([W, b])
    return params_final

  def forward_eval_single(self, params_final, t, x):
    """
    (Optional) this is a standard forward pass of a mlp. This is useful to speed up the performance during test time 
    """
    # concatenate coordinate and latent code
    x = np.append(x, t)

    # forward pass
    for ii in range(len(params_final) - 1):
        W, b = params_final[ii]
        x = jax.nn.relu(np.dot(W, x) + b)
    W, b = params_final[-1] # final layer
    out = np.dot(W, x) + b
    return out[0]
  forward_eval = jax.vmap(forward_eval_single, in_axes=(None, None, None, 0), out_axes=0)
