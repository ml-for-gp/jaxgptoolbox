import sys
sys.path.append('../../../')
import jaxgptoolbox as jgp
import numpy as np

# create a random voxel grid (inside outside)
x_res = 7
y_res = 9
z_res = 11
W = np.zeros((x_res,y_res,z_res))
for x in range(x_res):
    for y in range(y_res):
        for z in range(z_res):
            if np.sqrt(x**2+y**2+z**2) > 5:
                W[x,y,z] = 1


# create vertices
[bx,by,bz] = np.meshgrid(np.arange(x_res+1), np.arange(y_res+1), np.arange(z_res+1))
V = np.vstack((bx.flatten(),by.flatten(),bz.flatten())).T
print(V)

