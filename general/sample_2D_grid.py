import numpy as onp
import jax.numpy as np

def sample_2D_grid(resolution, low = 0, high = 1):
  idx = onp.linspace(low,high,num=resolution)
  x, y = onp.meshgrid(idx, idx)
  V = onp.concatenate((x.reshape((-1,1)), y.reshape((-1,1))), 1)
  return np.array(V)