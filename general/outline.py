import numpy as onp
import scipy 
import jax.numpy as np
def outline(F_jnp):
    """
    this function extract (unordered) boundary edges of a mesh

    Inputs
    F_jnp: |F|x3 jax numpy array of the face indices

    Outputs
    O: |E|x2 jax numpy array of unordered boundary edges

    Reference:
    this code is adapted from https://github.com/alecjacobson/gptoolbox/blob/master/mesh/outline.m
    """
    F = onp.array(F_jnp) # convert the numpy for efficiency
    nV = F.max()+1
    row = F.flatten()
    col = F[:,[1,2,0]].flatten()
    data = onp.ones(len(row), dtype=np.int32)
    A = scipy.sparse.csr_matrix((data, (row, col)), shape=(nV,nV)) # build directed adj matrix
    AA = A - A.transpose() # figure out edges that only have one half edge
    AA.eliminate_zeros()
    I,J,V = scipy.sparse.find(AA) # get the non-zeros
    O = np.array([I[V>0], J[V>0]]).T # construct the boundary edge list
    return O


# =====
# I switch to the gptoolbox version because a short prifiling show that the above version is 5x faster than the bottom version on a mesh with 1200 faces
# =====
# import numpy as np
# import jax

# def outline(F):
# 	'''
# 	OUTLINE compute the unordered outline edges of a triangle mesh

# 	Input:
# 	  F (|F|,3) numpy array of face indices
# 	Output:
# 	  O (|bE|,2) numpy array of boundary vertex indices, one edge per row
# 	'''

# 	# All halfedges
# 	he = np.stack((np.ravel(F[:,[1,2,0]]),np.ravel(F[:,[2,0,1]])), axis=1)

# 	# Sort hes to be able to find duplicates later
# 	# inds = np.argsort(he, axis=1)
# 	# he_sorted = np.sort(he, inds, axis=1)
# 	he_sorted = np.sort(he, axis=1)

# 	# Extract unique rows
# 	_,unique_indices,unique_counts = np.unique(he_sorted, axis=0, return_index=True, return_counts=True)

# 	# All the indices with only one unique count are the original boundary edges
# 	# in he
# 	O = he[unique_indices[unique_counts==1],:]

# 	return O
