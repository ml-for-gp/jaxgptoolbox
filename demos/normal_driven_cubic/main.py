import sys
sys.path.append('../../../')
import jaxgptoolbox as jgp

from jaxgptoolbox.differentiable.normalizerow import normalizerow
from jaxgptoolbox.differentiable.fit_rotations_cayley import fit_rotations_cayley
import numpy.matlib as matlib
import scipy
import scipy.sparse 

import jax
import jax.numpy as np
from jax import jit, value_and_grad
from jax.example_libraries import optimizers

import numpy as onp
import pickle
import tqdm

def spokes_rims(V,F):
    """
    build spokes and rims edge indices and edge weights

    Inputs:
    V: |V|x3 array of vertex list
    F: |F|x3 array of face list

    Outputs:
    Ek_all: |V| list of arrays, where Ek_all[v] = |Ek|x2 array of edge indices
    Wk_all: |V| list of arrays, where Wk_all[v] = |Ek| array of cotan weights
    """
    V2F = jgp.vertex_face_adjacency_list(F)
    C = jgp.cotangent_weights(V,F)

    # construct spokes and rims for each vertex
    F = onp.array(F)
    nV = V.shape[0]

    # find the max number of neighbors
    max_Nk = 0
    for kk in range(nV):
        len_Nk = len(V2F[kk])
        if len_Nk > max_Nk:
            max_Nk = len_Nk
    print("max #neighbors is: %d" % max_Nk)

    Ek_np = onp.zeros((nV,max_Nk*3,2), dtype=onp.int32)
    Wk_np = onp.zeros((nV,max_Nk*3), dtype=onp.float32)
    for kk in range(nV):
        # get neighbors
        Nk = V2F[kk]

        # construct edge list
        Ek0 = onp.concatenate((F[Nk,0], F[Nk,1], F[Nk,2]))
        Ek1 = onp.concatenate((F[Nk,1], F[Nk,2], F[Nk,0]))
        Ek = onp.concatenate((Ek0[:,None], Ek1[:,None]), axis = 1)
        # pad with the ghost vertex index so that Ek becomes a numpy array with size (max_Nk, 2)
        Ek_pad = onp.pad(Ek, pad_width = ((0, max_Nk*3 - Ek.shape[0]), (0, 0)), mode = 'constant', constant_values = nV) 
        Ek_np[kk,:,:] = Ek_pad

        # get all the cotan weights
        Wk = onp.concatenate((C[Nk,2],C[Nk,0],C[Nk,1]))
        # pad with 0 so that Wk becomes a numpy array with size (max_Nk,)
        Wk_pad = onp.pad(Wk, pad_width = ((0, max_Nk*3 - Wk.shape[0])), mode = 'constant', constant_values = 0) 
        Wk_np[kk,:] = Wk_pad

    return np.array(Ek_np), np.array(Wk_np)

def compute_target_normal_single(n):
    cIdx = np.argmax(np.abs(n))
    tar_n = np.zeros((3,), dtype=np.float32)
    tar_n = tar_n.at[cIdx].set(np.sign(n[cIdx]))
    return tar_n
compute_target_normals = jax.vmap(compute_target_normal_single, in_axes=(0), out_axes=0)

def normal_driven_energy_single(U,V,lam,n,tar_n,a,E,W):
    v_ghost = np.array([[0.,0.,0.]])
    Ug = np.concatenate((U, v_ghost), axis = 0)
    Vg = np.concatenate((V, v_ghost), axis = 0)

    dV = (Vg[E[:,1],:] - Vg[E[:,0],:]).T
    dU = (Ug[E[:,1],:] - Ug[E[:,0],:]).T

    # orthogonal procrustes
    S = (dV * W).dot(dU.T) + lam*a*n[:,None].dot(tar_n[None,:])

    # fit rotation
    R = fit_rotations_cayley(S)

    # compute loss
    RdV_dU = R.dot(dV) - dU
    Rn_tar_n = R.dot(n) - tar_n
    return np.trace((RdV_dU * W).dot(RdV_dU.T)) + lam*a*Rn_tar_n.dot(Rn_tar_n)
normal_driven_energy = jax.vmap(normal_driven_energy_single, in_axes=(None,None,None,0,0,0,0,0), out_axes=0)

# define loss function and update function
def loss(U,V,lam,N,tar_N,VA,Ek_all,Wk_all):
    loss = normal_driven_energy(U,V,lam,N,tar_N,VA,Ek_all,Wk_all)
    return loss.mean()

@jit
def update(epoch, opt_state, V,lam,N,tar_N,VA,Ek_all,Wk_all):
    U = get_params(opt_state)
    value, grads = value_and_grad(loss, argnums = 0)(U,V,lam,N,tar_N,VA,Ek_all,Wk_all)
    opt_state = opt_update(epoch, grads, opt_state)
    return value, opt_state

if __name__ == "__main__":
    # hyper parameters
    hyper_params = {
        "step_size": 1e-4,
        "num_epochs": 1000,
    }

    V,F = jgp.read_mesh("./spot.obj")
    N = jgp.vertex_normals(V,F)
    VA = jgp.vertex_areas(V,F)
    Ek_all, Wk_all = spokes_rims(V,F)
    tar_N = compute_target_normals(N)
    data = [V,F,N,tar_N,VA,Ek_all,Wk_all]

    U = V.copy()
    lam = 1.0

    # optimizer
    opt_init, opt_update, get_params = optimizers.adam(step_size=hyper_params["step_size"])
    opt_state = opt_init(U)

    # training
    loss_history = onp.zeros(hyper_params["num_epochs"])
    pbar = tqdm.tqdm(range(hyper_params["num_epochs"]))
    for epoch in pbar:
        loss_value, opt_state = update(epoch, opt_state, V,lam,N,tar_N,VA,Ek_all,Wk_all)
        loss_history[epoch] = loss_value
        pbar.set_postfix({"loss": loss_value})

    U = get_params(opt_state)
    jgp.writeOBJ("opt.obj", U,F)



