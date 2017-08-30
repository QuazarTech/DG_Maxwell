#! /usr/bin/env python3

import arrayfire as af
import numpy as np
from app import lagrange
from utils import utils
from app import global_variables as gvar

def Li_Lp_xi(L_i_xi, L_p_xi):
    '''
    Broadcasts and multiplies two arrays of different ordering. Used in the
    calculation of A matrix.
    
    Parameters
    ----------
    L_i_xi : arrayfire.Array [1 N N 1]
             A 2D array :math:`L_i` obtained at LGL points calculated at
             the LGL nodes :math:`N_LGL`.
    
    L_p_xi : arrayfire.Array [N 1 N 1]
             A 2D array :math:`L_p` obtained at LGL points calculated at the
             LGL nodes :math:`N_LGL`.
    
    Returns
    -------    
    Li_Lp_xi : arrayfire.Array [N N N 1]
               Matrix of :math:`L_p (N_LGL) L_i (N_LGL)`.
    
    '''
    Li_Lp_xi = af.bcast.broadcast(utils.multiply, L_i_xi, L_p_xi)
    
    return Li_Lp_xi


def mappingXiToX(x_nodes, xi):
    '''
    Maps points in :math: `\\xi` space to :math:`x` space using the formula
    :math:  `x = \\frac{1 - \\xi}{2} x_0 + \\frac{1 + \\xi}{2} x_1`
    
    Parameters
    ----------
    
    x_nodes : arrayfire.Array
              Element nodes.
    
    xi      : numpy.float64
              Value of :math: `\\xi`coordinate for which the corresponding
              :math: `x` coordinate is to be found.
.
    
    Returns
    -------
    x : arrayfire.Array
        :math: `x` value in the element corresponding to :math:`\\xi`.
    '''
    N_0 = (1 - xi) / 2
    N_1 = (1 + xi) / 2
    
    N0_x0 = af.bcast.broadcast(utils.multiply, N_0, x_nodes[0])
    N1_x1 = af.bcast.broadcast(utils.multiply, N_1, x_nodes[1])
    
    x = N0_x0 + N1_x1
    
    return x


def dx_dxi_numerical(x_nodes, xi):
    '''
    Differential :math: `\\frac{dx}{d \\xi}` calculated by central differential
    method about xi using the mappingXiToX function.
    
    Parameters
    ----------
    
    x_nodes : arrayfire.Array
              Contains the nodes of elements
    
    xi      : float
              Value of :math: `\\xi`
    
    Returns
    -------
    dx_dxi : arrayfire.Array
             :math:`\\frac{dx}{d \\xi}`. 
    '''
    dxi = 1e-7
    x2 = mappingXiToX(x_nodes, xi + dxi)
    x1 = mappingXiToX(x_nodes, xi - dxi)
    
    dx_dxi = (x2 - x1) / (2 * dxi)
    
    return dx_dxi


def dx_dxi_analytical(x_nodes, xi):
    '''
    The analytical result for :math:`\\frac{dx}{d \\xi}` for a 1D element is
    :math: `\\frac{x_1 - x_0}{2}
    Parameters
    ----------
    x_nodes : arrayfire.Array
              An array containing the nodes of an element.
    
    Returns
    -------
    analytical_dx_dxi : arrayfire.Array
                        The analytical solution of
                        \\frac{dx}{d\\xi} for an element.
    
    '''
    analytical_dx_dxi = (x_nodes[1] - x_nodes[0]) / 2
    
    return analytical_dx_dxi

def A_matrix():
    '''
    Calculates A matrix whose elements :math:`A_{p i}` are given by
    :math: `A_{p i} &= \\int^1_{-1} L_p(\\xi)L_i(\\xi) \\frac{dx}{d\\xi}`
    These elements are to be arranged in an :math:`N \times N` array with p
    varying from 0 to N - 1 along the rows and i along the columns.
    The integration is carried out using Gauss-Lobatto quadrature.
    
    Full description of the algorithm can be found here-
    https://cocalc.com/projects/1b7f404c-87ba-40d0-816c-2eba17466aa8/files/
    PM_2_5/wave_equation/documents/wave_equation_report
    /A_matrix.pdf?session=default
    
    Returns
    -------
    A_matrix : arrayfire.Array
               The value of integral of product of lagrange basis functions
               obtained by LGL points, using Gauss-Lobatto quadrature method
               using :math: `N_LGL` points. 
    '''    
    
    x_tile           = af.transpose(af.tile(gvar.xi_LGL, 1, gvar.N_LGL))
    power            = af.flip(af.range(gvar.N_LGL))
    power_tile       = af.tile(power, 1, gvar.N_LGL)
    x_pow            = af.arith.pow(x_tile, power_tile)
    lobatto_weights  = gvar.lobatto_weights
    
    lobatto_weights_tile = af.tile(af.reorder(lobatto_weights, 1, 2, 0),\
                                                gvar.N_LGL, gvar.N_LGL)
    
    index = af.range(gvar.N_LGL)
    L_i   = af.blas.matmul(gvar.lBasisArray[index], x_pow)
    L_p   = af.reorder(L_i, 0, 2, 1)
    L_i   = af.reorder(L_i, 2, 0, 1)
    
    dx_dxi      = dx_dxi_numerical((gvar.x_nodes),gvar.xi_LGL)
    dx_dxi_tile = af.tile(dx_dxi, 1, gvar.N_LGL, gvar.N_LGL)
    Li_Lp_array = Li_Lp_xi(L_p, L_i)
    L_element   = (Li_Lp_array * lobatto_weights_tile * dx_dxi_tile)
    A_matrix    = af.sum(L_element, dim = 2)
    
    return A_matrix


def flux_x(u):
    '''
    A function which calcultes and returns the value of flux for a given
    wave function u. :math:`f(u) = c u^k`
    
    Parameters
    ----------
    u : arrayfire.Array [N 1 1 1]
        A 1-D array which contains the value of wave function.
    
    Returns
    -------
    flux : arrayfire.Array
           The value of the flux for given u.
    '''
    flux_x = gvar.c * u
    
    return flux_x


def volumeIntegralFlux(u):
    '''
    A function to calculate the volume integral of flux in the wave equation.
    :math:`\\int_{-1}^1 f(u) \\frac{d L_p}{d\\xi} d\\xi`
    This will give N values of flux integral as p varies from 0 to N - 1.
    
    This integral is carried out over an element with LGL nodes mapped onto it.
    
    Parameters
    ----------
    u : arrayfire.Array [N M 1 1]
        An N_LGL x N_Elements array containing the value of the
        wave function at the mapped LGL nodes in all the elements.
    
    Returns
    -------
    flux_integral : arrayfire.Array [N M 1 1]
                    A 1-D array of the value of the flux integral calculated
                    for various lagrange basis functions.
    '''
    
    dLp_xi          = gvar.dLp_xi
    lobatto_weights = af.transpose(gvar.lobatto_weights)
    dLp_xi_weights  = af.broadcast(utils.multiply, lobatto_weights, dLp_xi)
    flux            = flux_x(u)
    flux_integral   = af.blas.matmul(dLp_xi_weights, flux)
    
    return flux_integral
