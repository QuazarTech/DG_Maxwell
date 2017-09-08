#! /usr/bin/env python3
import numpy as np
import arrayfire as af
from utils import utils
from app import global_variables as gvar


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


def lagrange_basis_function(lagrange_coeff_array):
    '''
    Funtion which calculates the value of lagrange basis functions over LGL
    nodes.
    
    Returns
    -------
    L_i    : arrayfire.Array [N 1 1 1]
             The value of lagrange basis functions calculated over the LGL
             nodes.
    '''
    xi_tile    = af.transpose(af.tile(gvar.xi_LGL, 1, gvar.N_LGL))
    power      = af.flip(af.range(gvar.N_LGL))
    power_tile = af.tile(power, 1, gvar.N_LGL)
    xi_pow     = af.arith.pow(xi_tile, power_tile)
    index      = af.range(gvar.N_LGL)
    L_i        = af.blas.matmul(lagrange_coeff_array[index], xi_pow)
    
    return L_i


def dLp_xi_LGL(lagrange_coeff_array):
    '''
    Function to calculate :math: `\\frac{d L_p(\\xi_{LGL})}{d\\xi}`
    as a 2D array of :math: `L_i' (\\xi_{LGL})`. Where i varies along rows
    and the nodes vary along the columns.
    
    Returns
    -------
    dLp_xi        : arrayfire.Array [N N 1 1]
                    A 2D array :math: `L_i (\\xi_p)`, where i varies
                    along dimension 1 and p varies along second dimension.
    '''
    differentiation_coeffs = (af.transpose(af.flip(af.tile\
                             (af.range(gvar.N_LGL), 1, gvar.N_LGL)))
                             * lagrange_coeff_array)[:, :-1]
    
    nodes_tile         = af.transpose(af.tile(gvar.xi_LGL, 1, gvar.N_LGL - 1))
    power_tile         = af.flip(af.tile(af.range(gvar.N_LGL - 1), 1, gvar.N_LGL))
    nodes_power_tile   = af.pow(nodes_tile, power_tile)
    
    dLp_xi = af.blas.matmul(differentiation_coeffs, nodes_power_tile)
    
    return dLp_xi