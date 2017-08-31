#! /usr/bin/env python3
import numpy as np
import arrayfire as af
from utils import utils
from app import global_variables as gvar

def LGL_points(N):
    '''
    Returns the :math: `N` Legendre-Gauss-Laguere points which are
    the roots of the equation
    :math::
        (1 - x^2)L'_N = 0
    
    Parameters
    ----------
    N : int
        Number of LGL-points to be generated.
        2 < N < 16
    
    Returns
    -------
    lgl : arrayfire.Array
          An array of :math: `N` LGL points.
    '''
    if N > 16 or N < 2:
        print('Skipping! This function can only return from ',
              '2 to 16 LGL points.')
    
    n = N - 2

    lgl = af.Array(gvar.LGL_list[n])
    return lgl


def lagrange_basis_coeffs(x):    
    '''
    A function to get the coefficients of the Lagrange basis polynomials for
    a given set of x nodes.
    
    This function calculates the Lagrange basis
    polynomials by this formula:
    :math::
    `L_i = \\prod_{m = 0, m \\notin i}^{N - 1}\\frac{(x - x_m)}{(x_i - x_m)}`

    Parameters
    ----------
    x : numpy.array
        A numpy array consisting of the :math: `x` nodes using which the
        lagrange basis functions need to be evaluated.

    Returns
    -------
    lagrange_basis_poly : numpy.ndarray
                          A :math: `N \\times N` matrix containing the
                          coefficients of the Lagrange basis polynomials such
                          that :math:`i^{th}` lagrange polynomial will be the
                          :math:`i^{th}` row of the matrix.
    '''
    X = np.array(x)
    lagrange_basis_poly = np.zeros([X.shape[0], X.shape[0]])
    
    for j in np.arange(X.shape[0]):
        lagrange_basis_j = np.poly1d([1])
        
        for m in np.arange(X.shape[0]):
            if m != j:
                lagrange_basis_j *= np.poly1d([1, -X[m]]) \
                                    / (X[j] - X[m])
        lagrange_basis_poly[j] = lagrange_basis_j.c
    
    return lagrange_basis_poly


def lagrange_basis(i, x):
    '''
    Calculates the value of the :math: `i^{th}` Lagrange basis (calculated
    using the :math:`\\xi_{LGL}` points) at the :math: `x` coordinates.
    
    Parameters
    ----------
    
    i : int
        The Lagrange basis which is to be calculated.
    
    x : arrayfire.Array
        The coordinates at which the `i^{th}` Lagrange polynomial is to be
        calculated.
    
    Returns
    -------
    
    l_xi_j : arrayfire.Array
             Array of values of the :math: `i^{th}` Lagrange basis
             calculated at the given :math: `x` coordinates.
    '''
    x_tile      = af.transpose(af.tile(x, 1, gvar.N_LGL))
    power       = utils.linspace((gvar.N_LGL-1), 0, gvar.N_LGL)
    power_tile  = af.tile(power, 1, x.shape[0])
    x_pow       = af.arith.pow(x_tile, power_tile)
    l_xi_j      = af.blas.matmul(gvar.lBasisArray[i], x_pow)
    
    return l_xi_j
