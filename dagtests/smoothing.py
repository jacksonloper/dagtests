'''
Functions for combining p values
'''

import numpy as np
import scipy as sp
import scipy.stats
import networkx

def simes(p_values, beta=None):
    '''Implements the generalized Simes p-value. beta is an optional reshaping function.'''
    p_sorted = p_values[np.argsort(p_values)]
    if beta is None:
        beta = lambda x: x
    return (p_sorted * p_sorted.shape[0] / beta(np.arange(1,p_sorted.shape[0]+1))).min()

def fisher(p_values):
    '''
    Implements Fisher's method for combining p-values.
    '''

    # Convert to numpy
    p_values = np.require(p_values)

    # Check for hard zeroes
    if p_values.min()<=0:
        return 0

    # Get the number of p-values on the axis of interest
    N = np.prod(p_values.shape)

    # Fisher merge
    results = sp.stats.chi2.sf(-2 * np.log(p_values).sum(), 2*N)

    return results

def conservative_stouffer(p_values):
    '''
    Stouffer that gives superuniform result for superuniform p_values
    with any gaussian copula
    '''
    p_values=np.require(p_values)
    Z_values = sp.stats.norm.ppf(p_values)
    meanZ = np.mean(Z_values)

    if meanZ > 0:
        return 1.0
    else:
        return sp.stats.norm.cdf(meanZ)

def stouffer(p_values):
    '''
    stouffer method for combining p values
    '''
    p_values=np.require(p_values)
    Z_values = sp.stats.norm.ppf(p_values)
    n=np.prod(Z_values.shape)

    return sp.stats.norm.cdf(np.sum(Z_values)/np.sqrt(n))

def smooth_dag_at_distance(adj_matrix, p_values, smoother,distance=np.inf,
            shortest_path_precomputation=None,**smootherkwargs):
    '''
    apply a combining-p-value method to a dag
    '''
    p_values=np.require(p_values)
    g = networkx.DiGraph(adj_matrix)

    if shortest_path_precomputation is None:
        shortest_path_precomputation={x:y for (x,y) in networkx.shortest_path_length(g)}


    q_values = np.zeros_like(p_values)
    for node in range(len(p_values)):
        spp=shortest_path_precomputation[node]
        descendants=[x for x in spp if spp[x]<=distance]
        descendants.append(node)
        q_values[node] = smoother(p_values[descendants],**smootherkwargs)
    return q_values


























