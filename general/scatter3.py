import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def scatter3(P):
  """
  A simple wrapper of matplotlib scatter plot in 3D
  """
  fig = plt.figure()
  ax = plt.axes(projection='3d')
  ax.scatter(P[:,0], P[:,1], P[:,2])
  ax.axis('equal')