#! /usr/bin/env python3

from os import sys

import arrayfire as af
af.set_backend('opencl')
import numpy as np
from scipy import special as sp
from app import lagrange
from app import wave_equation
from utils import utils



N_LGL      = 8 
N_Elements = 10

def LGL_points(N = N_LGL):
    '''
    Used to obtain the Legendre-Gauss-Lobatto points from the list
    of LGL points.
    '''
    xi                 = np.poly1d([1, 0])
    legendre_N_minus_1 = N * (xi * sp.legendre(N - 1) - sp.legendre(N))
    lgl_points         = legendre_N_minus_1.r
    lgl_points.sort()
    lgl_points         = af.np_to_af_array(lgl_points)
    
    return lgl_points 


def lobatto_weight_function(n, x):
    '''
    Calculates and returns the weight function for an index :math:`n`
    and points :math: `x`.
    
    :math::
        `w_{n} = \\frac{2 P(x)^2}{n (n - 1)}`,
        Where P(x) is $ (n - 1)^th $ index.
    
    Parameters
    ----------
    n : int
        Index for which lobatto weight function
    
    x : arrayfire.Array
        1D array of points where weight function is to be calculated.
    
    .. lobatto weight function -
    https://en.wikipedia.org/wiki/
    Gaussian_quadrature#Gauss.E2.80.93Lobatto_rules
    
    Returns
    -------
    lobatto_weights : arrayfire.Array
                      An array of lobatto weight functions for
                      the given :math: `x` points and index.
    
    '''
    P = sp.legendre(n - 1)
    
    lobatto_weights = (2 / (n * (n - 1)) / (P(x))**2)

    return lobatto_weights  

def dLp_xi_LGL():
    '''
    Function to calculate :math: `\\frac{d L_p(\\xi_{LGL})}{d\\xi}`
    as a 2D array of :math: `L_i (\\xi_{LGL})`. Where i varies along rows
    and the nodes vary along the columns.
    
    Returns
    -------
    dLp_xi        : arrayfire.Array [N N 1 1]
                    A 2D array :math: `L_i (\\xi_p)`, where i varies
                    along dimension 1 and p varies along second dimension.
    '''
    differentiation_coeffs = (af.transpose(af.flip(af.tile(af.range(N_LGL),\
                                         1, N_LGL))) * lBasisArray)[:, :-1]
    
    nodes_tile         = af.transpose(af.tile(xi_LGL, 1, N_LGL - 1))
    power_tile         = af.flip(af.tile\
                                    (af.range(N_LGL - 1), 1, N_LGL))
    nodes_power_tile   = af.pow(nodes_tile, power_tile)
    
    dLp_xi = af.blas.matmul(differentiation_coeffs, nodes_power_tile)
    
    return dLp_xi






xi_LGL          = LGL_points()
eta_LGL         = LGL_points()
lobatto_weights = af.np_to_af_array(lobatto_weight_function(N_LGL, xi_LGL))
test_nodes      = af.interop.np_to_af_array(np.array([[0., 2], [0, 1], [0, 0], \
                                                    [1, 0], [2, 0], [2, 1],\
                                                    [2, 2], [1, 2]]))

x_nodes = test_nodes[:, 0]
y_nodes = test_nodes[:, 1]

lagrange_coeff_array = lagrange.lagrange_basis_coeffs(xi_LGL)
lagrange_value_array = lagrange.lagrange_basis_function()
