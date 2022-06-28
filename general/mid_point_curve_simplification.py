import numpy as np
from .ordered_outline import ordered_outline
from .remove_unreferenced import remove_unreferenced

def mid_point_curve_simplification(V,O,tarE):
    """
    this function simplify a single closed curve via collapsing the shortest edge

    Inputs
    V: |V|x3 vertex list
    O: |O|x2 (unordered) boundary edges 
    tarE: target number of edges in the simplified curve

    Outputs
    V: |Vc|x3 simplified vertex list
    O: tarEx3 simplified boundary curve

    Warning:
    - This only support single closed curve
    - This is a simple collapst which does not preserve geometry nor avoid collision
    """
    V = np.array(V)
    L = ordered_outline(O)
    E_list = np.array(L[0])
    E = np.array([E_list, np.roll(E_list,-1)]).T

    # mid point collapse ordered outline
    ECost = np.sqrt(np.sum((V[E[:,0],:] - V[E[:,1],:])**2,1)) # cost is outline edge lengths

    total_collapses = E.shape[0] - tarE
    num_collapses = 0

    while True:
        if num_collapses % 100 == 0:
            print("collapse progress %d / %d\n" % (num_collapses, total_collapses))

        # get the minimum cost (slow)
        # note: a faster version should use a priority queue
        e = np.argmin(ECost)

        # check if the edge is degenerated
        if E[e,0] == E[e,1]:
            E = np.delete(E,e,0)
            ECost = np.delete(ECost,e)
            continue
        
        # move vertex vi
        vi, vj = E[e,:]
        V[vi,:] = (V[vi,:] + V[vj,:]) / 2.

        # reconnect edges
        prev_e = (e-1) % E.shape[0]
        next_e = (e+1) % E.shape[0]
        E[next_e,0] = vi # keep E[e,0] and unreference vj

        # update edge costs
        ECost[prev_e] = np.sqrt(np.sum((V[E[prev_e,0],:] - V[E[prev_e,1],:])**2))
        ECost[next_e] = np.sqrt(np.sum((V[E[next_e,0],:] - V[E[next_e,1],:])**2))

        # post collapse update
        E = np.delete(E,e,0)
        ECost = np.delete(ECost,e)
        num_collapses += 1

        # stopping
        if num_collapses == total_collapses:
            break
    V,E,_ = remove_unreferenced(V,E)
    return V, E 