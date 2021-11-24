import scipy
import scipy.spatial

def knn_search(query_points, source_points, k):
    """
    KNNSEARCH finds the k nearnest neighbors of query_points in source_points

    Inputs:
        query_points: N-by-D numpy array of query points
        source_points: M-by-D numpy array existing points
        k: number of neighbors to return

    Output:
        dist: distance between the point in array1 with kNN
        NNIdx: nearest neighbor indices of array1
    """
    kdtree = scipy.spatial.cKDTree(source_points)
    dist, NNIdx = kdtree.query(query_points, k)
    return dist, NNIdx