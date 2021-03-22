'''
Implements the "all-parents" algorithm from Meijer and Goeman (2015)
for controlling FWER on a DAG.

MIT Licens
Jackson Loper & Wes Tansey
'''

import numpy as np
import networkx
from . import utils

def select(adj_matrix, p_vals, alpha):
    '''The selection procedure to test p-values with a DAG structure.
    adj_matrix: The adjacency matrix for determining the DAG structure.
                Entry X[i,j] = 1 if node i has an edge to child node j.

    p_vals: the p-values for each node.

    alpha: the type I (FWER) error threshold.
    '''

    # Construct the network from the adjacency matrix
    g = networkx.DiGraph(adj_matrix)
    rejected = np.zeros(len(p_vals), dtype=bool)

    while True:
        # Apply the weighting function from Meijer and Goeman
        weights = get_weights(g, rejected)

        # Apply the selection function from Meijer and Goeman at the alpha level
        updated = False
        for node in networkx.topological_sort(g):
            # Skip this node if it's already been rejected
            if rejected[node]:
                continue

            # Only reject nodes whose parents have all been rejected
            parents = utils.parents_of(adj_matrix,node)
            if len(parents) > 0 and not np.all(rejected[parents]):
                continue

            # Reject nodes at or below the weighted alpha level
            if p_vals[node] <= alpha*weights[node]:
                updated = True
                rejected[node] = True

        # Stop if we've converged
        if not updated:
            break

    return rejected

def get_weights(g, rejected):
    '''
    Compute weights according to all-parents method of MG
    '''

    # Initialize the weights
    weights = np.zeros(g.number_of_nodes())

    # Get the |V| leaf nodes
    leaves = [v for v, d in g.out_degree() if d == 0 and not rejected[v]]

    if len(leaves) == 0:
        return weights

    # Set all the leaf nodes to have 1/|V| weight
    weights[leaves] = 1/len(leaves)

    # Get the nodes in toplogical order from the bottom of the graph
    ordered_inds = np.array([idx for idx in networkx.topological_sort(g)])[::-1]

    # Update the nodes in topological order
    for node in ordered_inds:
        # Ignore nodes that have already been rejected
        if rejected[node]:
            continue

        # Get all unrejected parents of the current node
        parents = [w for w in utils.parents_of(g, node) if not rejected[w]]

        # Root nodes have no parents
        if len(parents) == 0:
            continue

        # Distribute the current node weight equally amongst its unrejected parents
        weights[parents] += weights[node] / len(parents)

    return weights

def select_fdx(adj_matrix, p_vals, alpha, gamma):
    '''
    FDX extension of MG
    '''

    # get fwer selections
    selections = select(adj_matrix, p_vals, alpha)

    # Get the number of selections
    nfwer = np.sum(selections)

    # calculate the magic number
    magic_number = int(np.floor(nfwer * gamma/(1-gamma)))

    # Get the nodes in toplogical order from the top of the graph
    # (resolve ambiguities by going for the low p-values first)
    g = networkx.DiGraph(adj_matrix)
    ordered_inds = np.array(list(networkx.lexicographical_topological_sort(
        g, key=lambda i:p_vals[i])))

    # only care about ones which we haven't already rejected
    ordered_inds = ordered_inds[~selections]

    # get the top magic-number of them, add 'em to the list!
    selections[ordered_inds[:magic_number]]=True

    # done
    return selections
