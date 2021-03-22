'''
some helpful things for handling networks
'''

import numpy as np
import networkx

def parents_of(adj_matrix, node):
    '''
    given an adjacency matrix or a graph and a node,
    finds all parents of node
    '''
    if isinstance(adj_matrix,np.ndarray):
        return np.where(adj_matrix[:,node])[0]
    elif isinstance(adj_matrix,networkx.DiGraph):
        return np.array([x[0] for x in adj_matrix.in_edges(node)])
    else:
        raise NotImplementedError(type(adj_matrix))
