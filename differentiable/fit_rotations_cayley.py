import jax.numpy as np

def fit_rotations_cayley(S, R = np.eye(3)):
    """
    given a cross-covariance matrix S, this function outputs the closest rotation R. This method is based on "Fast Updates for Least-Squares Rotational Alignment" by Zhang et al 2021

    Input:
    S: 3x3 array of the corss covariance matrix
    R: (optional) 3x3 initial rotation matrix (default identity)

    Output
    R: rotation matrix which maximize np.trace(S@R)

    Warning:
    - I haven't add a input check to check whether S is a valid crosss covariance matrix, if not a valid  crosss covariance matrix, this method will not output rotation matrix
    - I haven't implemented a stopping criteria, I have to figure out how to imeplement that in JAX
    """
    for iter in range(3): # empirically, 3 iterations seem enough
        MM = S @ R
        z = cayley_step(MM)
        s = z.dot(z)
        z_col = np.expand_dims(z, 1)
        zzT = z_col @ z_col.T
        Z = np.array([[0., -z[2], z[1]], [z[2], 0., -z[0]], [-z[1], z[0], 0.]])
        R = 1./(1.+s) * R @ ((1.-s)*np.eye(3) + 2*zzT + 2*Z) 
        # if s < 1e-8:
            # return R
    return R

def cayley_step(M):
    m = np.array([M[1,2]-M[2,1], M[2,0]-M[0,2], M[0,1]-M[1,0]])
    i,j,k = 0,1,2
    two_lam0 = 2*M[i,i] + np.abs(M[i,j]+M[j,i]) + np.abs(M[i,k]+M[k,i])
    i,j,k = 1,2,0
    two_lam1 = 2*M[i,i] + np.abs(M[i,j]+M[j,i]) + np.abs(M[i,k]+M[k,i])
    i,j,k = 2,0,1
    two_lam2 = 2*M[i,i] + np.abs(M[i,j]+M[j,i]) + np.abs(M[i,k]+M[k,i])
    two_lam = np.maximum(np.maximum(two_lam0, two_lam1), two_lam2)
    t = np.trace(M)
    gs = np.maximum(t, two_lam - t)
    H = M + M.T - (t + np.sqrt(gs*gs + m.dot(m)))*np.eye(3)

    h_inv = 1. / np.linalg.det(H)
    d0 = np.linalg.det( np.array([-m, H[:,1], H[:,2]]) )
    d1 = np.linalg.det( np.array([H[:,0], -m, H[:,2]]) )
    d2 = np.linalg.det( np.array([H[:,0], H[:,1], -m]) )
    return h_inv * np.array([d0, d1, d2])