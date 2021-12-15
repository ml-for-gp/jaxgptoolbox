import jax.numpy as jnp
import numpy as np

def writeOBJ(fileName,V,F):
	"""
    WRITEOBJ write .obj file

    Input:
      filepath a string of mesh file path
      V (|V|,3) tensor of vertex positions
	  	F (|F|,3) tensor of face indices
		Output:
			an .obj file
    """
	f = open(fileName, 'w')
	V = np.array(V)
	F = np.array(F)
	for ii in range(V.shape[0]):
		string = 'v ' + str(V[ii,0]) + ' ' + str(V[ii,1]) + ' ' + str(V[ii,2]) + '\n'
		f.write(string)
	Ftemp = F + 1
	for ii in range(F.shape[0]):
		string = 'f ' + str(Ftemp[ii,0]) + ' ' + str(Ftemp[ii,1]) + ' ' + str(Ftemp[ii,2]) + '\n'
		f.write(string)
	f.close()