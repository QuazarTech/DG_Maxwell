#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import arrayfire as af
from scipy import special as sp

def LGL_points(N):
    '''
    Calculates : math:`N` Legendre-Gauss-Lobatto (LGL) points.
    LGL points are the roots of the polynomial 
    :math:`(1 - \\xi^2) P_{n - 1}'(\\xi) = 0`
    Where :math:`P_{n}(\\xi)` are the Legendre polynomials.
    This function finds the roots of the above polynomial.

    Parameters
    ----------
    N : int
        Number of LGL nodes required
    
    Returns
    -------
    lgl : np.ndarray [N]
          The Lagrange-Gauss-Lobatto Nodes.
                          
    See `link`_

    .. _link: https://goo.gl/KdG2Sv
    '''
    
    xi                 = np.poly1d([1, 0])
    legendre_N_minus_1 = N * (xi * sp.legendre(N - 1) - sp.legendre(N))
    lgl_points         = legendre_N_minus_1.r
    lgl_points.sort()
    
    return lgl_points
