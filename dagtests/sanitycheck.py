'''
functions for making sure we're not screwing up
'''

import networkx
import numpy as np

def sanity_check(adj_matrix,selections):
    '''
    Checks that the selections are consistent with the adjacency
    matrix, i.e. that

    - if selections[v] and w is an ancestor of v, then selections[w].
    '''

    g = networkx.DiGraph(adj_matrix)
    ordered_inds = np.array(networkx.lexicographical_topological_sort(
        g, key={i:s for (i,s) in enumerate(selections)}))

    selections_reordered=selections[ordered_inds]
    nrej=np.sum(selections)

    return selections_reordered[:nrej].all()
