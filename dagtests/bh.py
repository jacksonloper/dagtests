'''
implements Benjamini-Hochberg
'''

import numpy as np

def benjamini_hochberg(p, fdr):
    '''Performs Benjamini-Hochberg multiple hypothesis testing on z at the
    given false discovery rate threshold.'''
    p = np.require(p)
    assert len(p.shape)==1, "p should be vector"

    selections=np.zeros(p.shape[0],dtype=bool)

    p_orders = np.argsort(p)[::-1]
    m = float(len(p_orders))
    for k, s in enumerate(p_orders):
        if p[s] <= (m-k) / m * fdr:
            selections[p_orders[k:]]=True
            return selections

    return selections
