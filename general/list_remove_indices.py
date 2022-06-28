def list_remove_indices(l, indices):
    """
    this method remove multiple elements from a list by indices

    inputs
    l: list
    indices: indices to be removed

    outouts
    l: change in place
    """
    indices = sorted(indices, reverse=True)
    for idx in indices:
        if idx < len(l):
            l.pop(idx)