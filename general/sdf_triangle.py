import numpy as onp
import jax.numpy as np

def sdf_triangle(p, p0 = onp.array([.2,.2]), p1 = onp.array([.8,.2]), p2 = onp.array([.5,.8])):
    """
    output the signed distance value of a triangle in 2D

    Inputs
    p: nx2 array of locations
    p0,p1,p2: locations of the triangle corners

    Outputs
    array of signed distance values at x
    """
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
