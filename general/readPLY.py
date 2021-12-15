import numpy as np
import jax.numpy as jnp

def readPLY(filepath):
    """
    READPLY read .ply file

    Input:
      filepath a string of mesh file path
    Output:
      V (|V|,3) numpy array of vertex positions
	  F (|F|,3) numpy array of face indices
    """
    nV = None
    nF = None
    nProperty = 0
    VPropertyCount = 3
    endHeader = False
    VIdx = 0
    FIdx = 0
    Vdtype = None
    with open(filepath, "r") as f:
        lines = f.readlines()
        for line in lines:
            if not endHeader:
                if "element vertex" in line:
                    for s in line.split():
                        if s.isdigit():
                            nV = int(s)
                if "element face" in line:
                    for s in line.split():
                        if s.isdigit():
                            nF = int(s)
                if ("property float" in line) or ("property double" in line):   
                    if "float" in line:
                        Vdtype = np.float32
                    elif "double" in line:
                        Vdtype = np.float64
                    nProperty += 1
                    if nProperty > 3:
                        VPropertyCount = 6
                if "end_header" in line:  
                    endHeader = True
                    if nV > 0:
                        V = np.zeros((nV,3), dtype = Vdtype)
                        if VPropertyCount is 6:
                            VN = np.zeros((nV,3), dtype = Vdtype)
                        else:
                            VN = None
                    else:
                        V = None
                        VN = None
                    
                    if nF > 0:
                        F = np.zeros((nF,3), dtype = np.int32)
                    else:
                        F = None
                    
            else:
                if VIdx < nV:
                    tmp = np.fromstring(line, dtype = Vdtype, count = VPropertyCount, sep=" ")
                    V[VIdx,:] = tmp[:3]
                    if VPropertyCount is 6:
                        VN[VIdx,:] = tmp[3:]
                    VIdx += 1
                elif (VIdx >= nV) and (FIdx < nF):
                    tmp = np.fromstring(line, dtype = np.int32, count = 4, sep=" ")
                    F[FIdx,:] = tmp[1:]
                    FIdx += 1
    if V is not None:
        V = jnp.array(V)
    if F is not None:
        F = jnp.array(F)
    if VN is not None:
        VN = jnp.array(VN)
    return V, F, VN