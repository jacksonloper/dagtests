'''
functions for generating simulated datasets which may
be used to benchmark various methods
'''

import numpy as np
import numpy.random as npr
import scipy as sp
import scipy.stats
import networkx
from . import utils

def measure_quality(true_signal,selections):
    '''
    Compute fwer for a given method of performing selections

    - true_signal -- (n,) binary array with one entry for each hypothesis
    - selections -- (ntrials,n) binary array with selections over many trials
    '''
    fwe_failure=np.sum((~true_signal[None,:]) & selections,axis=1)>0
    powers=np.sum((true_signal[None,:] & selections),axis=1) / np.sum(true_signal)
    return dict(fwer=np.mean(fwe_failure),power=np.mean(powers))


def simple_simulated_p_values(true_signal, alternative_mean,ntrials):
    '''
    samples some p values for
    - ntrials experiments
    - where true_signal has the set of nonnull nodes
    - nonnull nodes are sampled from normcdf(randn()-alternative mean)
    - null nodes are sampled from rand
    '''
    nnodes = len(true_signal)
    nsig=np.sum(true_signal)

    # sample p_values
    p_values=npr.rand(ntrials,nnodes)
    p_values[:,true_signal] = sp.stats.norm.cdf(npr.randn(ntrials,nsig)-alternative_mean)
    return p_values

def AR_nulls_simulated_p_values(adj_matrix,true_signal,alternative_mean,ntrials):
    '''
    samples some p values for
    - ntrials experiments
    - where true_signal has the set of nonnull nodes
    - nonnull nodes are sampled from normcdf(randn()-alternative mean)
    - null nodes are sampled from an AR process on the adj matrix
    '''
    adj_matrix=np.require(adj_matrix)
    nnodes=adj_matrix.shape[0]
    assert adj_matrix.shape==(nnodes,nnodes)
    nsig=np.sum(true_signal)
    nnull=np.sum(~true_signal)

    p_values=np.zeros((ntrials,nnodes))

    # sample nonnull p_values
    p_values[:,true_signal] = sp.stats.norm.cdf(npr.randn(ntrials,nsig)-alternative_mean)

    # sample nulls, using an autoregressive
    null_adj=adj_matrix[~true_signal][:,~true_signal]
    cov = autoregressive_gaussian_bayesnet(null_adj,1,1)
    R = correlationify(cov)
    Rch = np.linalg.cholesky(R)
    p_values[:,~true_signal] = sp.stats.norm.cdf((Rch @ npr.randn(nnull,ntrials)).T)

    return p_values

def combine_gaussians(muX, covX, muY, alpha, covY):
    '''
    let

    X   ~ N(muX,covX)
    Y|X ~ N(muY + alpha @ X, covY)

    This function returns mu,cov such that

    (X,Y) ~ N(mu,cov)

    Input
    - muX   (n,)
    - covX  (n,n)
    - muY   (m,)
    - alpha (m,n)
    - covY  (m,m)

    '''

    n = muX.shape[0]
    m = muY.shape[0]

    assert alpha.shape == (m, n)
    assert covX.shape == (n, n)
    assert covY.shape == (m, m)

    mu = np.r_[muX, muY+alpha@muX]

    cov = np.zeros((n+m, n+m))
    cov[:n, :n] = covX
    cov[n:, n:] = covY + alpha@covX@alpha.T
    cov[:n, n:] = covX @ alpha.T
    cov[n:, :n] = alpha@covX

    return mu, cov

def gaussian_bayesnet_to_classic_params(bayesnet):
    '''
    The input bayesnet should be formatted as a list.

    bayesnet[i] should be a 3-tuple containing
    - mu, scalar
    - sigma, scalar
    - alpha, a vector of size (i-1)

    This encodes a model of the form

    Z_i | Z_{0...i-1} ~ N(mu_i + alpha_i^T Z_{0...i-1}, sigma_i^2)

    Output is mu,sigma, global mean and variance for the whole process
    '''

    if len(bayesnet)==0:
        return np.zeros((0)), np.zeros((0,0))

    lmu, lsig, alpha = bayesnet[0]
    mu = np.array([lmu])
    cov = np.array([[lsig**2]])

    for lmu, lsig, alpha in bayesnet[1:]:
        mu, cov = combine_gaussians(mu, cov, np.array(
            [lmu]), alpha, np.array([[lsig**2]]))
    return mu, cov

def autoregressive_gaussian_bayesnet(adj, parentweight, selfweight):
    '''
    input
    -- adj, (n x n) edge matrix
    -- parentweight, scalar
    -- selfweight, scalar

    we are then interested in the process

    Z_i = parentweight*(average_{j is parents of i} Z_j) + selfweight*noise

    we return the covariance of Z under that model
    '''

    g = networkx.DiGraph(adj)

    nodes_used = []
    node_lookup = {}  # maps node identities to their position in nodes_used
    bayesnet = []

    for node in networkx.topological_sort(g):
        # add this node to our list of used nodes
        nodes_used.append(node)
        node_lookup[node] = len(nodes_used)-1

        # get parents node identities (in the original indexing)
        null_parents = utils.parents_of(g, node)
        n_null_parents = len(null_parents)

        if n_null_parents > 0:
            # get the parents in the indexing of nodes_used
            null_parents_reindexed = [node_lookup[x] for x in null_parents]

            # create corresponding weight for averaging those parents
            alpha = np.zeros(len(nodes_used)-1)
            alpha[null_parents_reindexed] = parentweight/n_null_parents
        else:
            alpha = np.zeros(len(nodes_used)-1)

        bayesnet.append((0.0, selfweight, alpha[None, :]))

    _,cov=gaussian_bayesnet_to_classic_params(bayesnet)

    cov=cov[nodes_used][:,nodes_used]

    return cov

def correlationify(x):
    '''
    return a covariance matrix into a correlation matrix
    '''
    x = np.require(x)
    n = x.shape[0]
    assert x.shape == (n, n)
    R = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            R[i, j] = x[i, j]/np.sqrt(x[i, i]*x[j, j])
    return R
