import numpy as onp
import jax.numpy as np

def sample_2D_grid(resolution):
  idx = onp.linspace(0,1,num=resolution)
  x, y = onp.meshgrid(idx, idx)
  V = onp.concatenate((x.reshape((-1,1)), y.reshape((-1,1))), 1)
  return np.array(V)