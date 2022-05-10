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
    Lipschitz weight normalization based on the L-infinity norm (see Eq.9 in [Liu et al 2022])
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
    This function computes the Lipschitz regularization Eq.7 in the [Liu et al 2022] 
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

def star_sdf(x, r = 0.22):
  # reference: https://iquilezles.org/www/articles/distfunctions2d/distfunctions2d.htm

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

def circle_sdf(x, r = 0.282, center = np.array([0.5,0.5])):
  dx = x - center
  return np.sqrt(np.sum((dx)**2, axis = 1)) - r

def cross_sdf(p, bx=0.35, by=0.12, r=0.):
  # reference: https://iquilezles.org/www/articles/distfunctions2d/distfunctions2d.htm

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

def triangle_sdf(p, p0 = onp.array([.2,.2]), p1 = onp.array([.8,.2]), p2 = onp.array([.5,.8])):
  p = onp.array(p)
  e0 = p1 - p0
  e1 = p2 - p1
  e2 = p0 - p2
  v0 = p - p0
  v1 = p - p1
  v2 = p - p2
  pq0 = v0 - e0[None,:] * onp.clip( v0.dot(e0) / e0.dot(e0), 0.0, 1.0 )[:,None]
  pq1 = v1 - e1[None,:] * onp.clip( v1.dot(e1) / e1.dot(e1), 0.0, 1.0 )[:,None]
  pq2 = v2 - e2[None,:] * onp.clip( v2.dot(e2) / e2.dot(e2), 0.0, 1.0 )[:,None]
  s = onp.sign( e0[0]*e2[1] - e0[1]*e2[0] )

  pq0pq0 = onp.sum(pq0*pq0, 1)
  d0 = onp.array([pq0pq0, s*(v0[:,0]*e0[1]-v0[:,1]*e0[0])]).T
  pq1pq1 = onp.sum(pq1*pq1, 1)
  d1 = onp.array([pq1pq1, s*(v1[:,0]*e1[1]-v1[:,1]*e1[0])]).T
  pq2pq2 = onp.sum(pq2*pq2, 1)
  d2 = onp.array([pq2pq2, s*(v2[:,0]*e2[1]-v2[:,1]*e2[0])]).T
  d = onp.minimum(onp.minimum(d0, d1),d2)
  out = -onp.sqrt(d[:,0]) * onp.sign(d[:,1])
  return np.array(out)
