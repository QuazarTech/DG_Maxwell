#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy import special as sp
import arrayfire as af

from dg_maxwell import utils
from dg_maxwell import params

af.set_backend(params.backend)
af.set_device(params.device)

def LGL_points(N):
    '''
    Calculates : math: `N` Legendre-Gauss-Lobatto (LGL) points.
    LGL points are the roots of the polynomial

    :math: `(1 - \\xi ** 2) P_{n - 1}'(\\xi) = 0`

    Where :math: `P_{n}(\\xi)` are the Legendre polynomials.
    This function finds the roots of the above polynomial.

    Parameters
    ----------

    N : int
        Number of LGL nodes required

    Returns
    -------

    lgl : arrayfire.Array [N 1 1 1]
          The Lagrange-Gauss-Lobatto Nodes.

    **See:** `document`_
    .. _document: https://goo.gl/KdG2Sv

    '''
    xi                 = np.poly1d([1, 0])
    legendre_N_minus_1 = N * (xi * sp.legendre(N - 1) - sp.legendre(N))
    lgl_points         = legendre_N_minus_1.r
    lgl_points.sort()
    lgl_points         = af.np_to_af_array(lgl_points)

    return lgl_points



def lobatto_weights(n):
    '''
    Calculates and returns the weight function for an index n
    and points x.


    Parameters
    ----------
    n : int
        Lobatto weights for n quadrature points.


    Returns
    -------
    Lobatto_weights : arrayfire.Array
                      An array of lobatto weight functions for
                      the given x points and index.

    **See:** Gauss-Lobatto weights Wikipedia `link`_.

    .. _link: https://goo.gl/kYqTyK
    **Examples**

    lobatto_weight_function(4) returns the Gauss-Lobatto weights
    which are to be used with the Lobatto nodes 'LGL_points(4)'
    to integrate using Lobatto quadrature.
    '''
    xi_LGL = LGL_points(n)

    P = sp.legendre(n - 1)

    Lobatto_weights = (2 / (n * (n - 1)) / (P(xi_LGL))**2)
    Lobatto_weights = af.np_to_af_array(Lobatto_weights)

    return Lobatto_weights


def gauss_nodes(n):
    '''
    Calculates :math: `N` Gaussian nodes used for Integration by
    Gaussia quadrature.
    Gaussian node :math: `x_i` is the `i^{th}` root of
    :math: `P_n(\\xi)`
    Where :math: `P_{n}(\\xi)` are the Legendre polynomials.

    Parameters
    ----------

    n : int
        The number of Gaussian nodes required.

    Returns
    -------

    gauss_nodes : numpy.ndarray
                  The Gauss nodes :math: `x_i`.

    **See:** A Wikipedia article about the Gauss-Legendre quadrature `here`_

    .. _here: https://goo.gl/9gqLpe

    '''
    legendre = sp.legendre(n)
    gauss_nodes = legendre.r
    gauss_nodes.sort()

    return gauss_nodes


def gaussian_weights(N):
    '''
    Returns the gaussian weights :math:`w_i` for :math:`N` Gaussian Nodes
    at index :math:`i`. They are given by

    .. math:: w_i = \\frac{2}{(1 - x_i^2) P'n(x_i)}

    Where :math:`x_i` are the Gaussian nodes and :math:`P_{n}(\\xi)`
    are the Legendre polynomials.

    Parameters
    ----------

    N : int
        Number of Gaussian nodes for which the weight is to be calculated.


    Returns
    -------

    gaussian_weight : arrayfire.Array [N_quad 1 1 1]
                      The gaussian weights.
    '''
    index = np.arange(N) # Index `i` in `w_i`, varies from 0 to N_quad - 1

    gaussian_nodes = gauss_nodes(N)
    gaussian_weight  = 2 / ((1 - (gaussian_nodes[index]) ** 2) *\
                       (np.polyder(sp.legendre(N))(gaussian_nodes[index])) ** 2)

    gaussian_weight = af.np_to_af_array(gaussian_weight)

    return gaussian_weight


def lagrange_polynomial_coeffs(x):
    '''
    This function doesn't use poly1d. It calculates the coefficients of the
    Lagrange basis polynomials.




    A function to get the analytical form and the coefficients of
    Lagrange basis polynomials evaluated using x nodes.

    It calculates the Lagrange basis polynomials using the formula:

    .. math:: \\
        L_i = \\prod_{m = 0, m \\notin i}^{N - 1}\\frac{(x - x_m)}{(x_i - x_m)}

    Parameters
    ----------

    x : numpy.array [N_LGL 1 1 1]
        Contains the :math: `x` nodes using which the
        lagrange basis functions need to be evaluated.

    Returns
    -------

    lagrange_basis_coeffs : numpy.ndarray
                            A :math: `N \\times N` matrix containing the
                            coefficients of the Lagrange basis polynomials such
                            that :math:`i^{th}` lagrange polynomial will be the
                            :math:`i^{th}` row of the matrix.

    '''
    X = np.array(x)
    lagrange_basis_poly   = []
    lagrange_basis_coeffs = af.np_to_af_array(np.zeros([X.shape[0], X.shape[0]]))
    
    for j in np.arange(X.shape[0]):
        lagrange_basis_k = af.np_to_af_array(np.array([1.]))
        
        for m in np.arange(X.shape[0]):
            if m != j:
                lagrange_basis_k = af.convolve1(lagrange_basis_k,\
                        af.np_to_af_array(np.array([1, -X[m]])/ (X[j] - X[m])),\
                                                   conv_mode=af.CONV_MODE.EXPAND)
        lagrange_basis_coeffs[j] = af.transpose(lagrange_basis_k)
    
    return lagrange_basis_coeffs


def lagrange_polynomials(x):    
    '''
    A function to get the analytical form and the coefficients of
    Lagrange basis polynomials evaluated using x nodes.
    
    It calculates the Lagrange basis polynomials using the formula:
    
    .. math:: \\
        L_i = \\prod_{m = 0, m \\notin i}^{N - 1}\\frac{(x - x_m)}{(x_i - x_m)}
        
    Parameters
    ----------
    
    x : numpy.array [N_LGL 1 1 1]
        Contains the :math: `x` nodes using which the
        lagrange basis functions need to be evaluated.
        
    Returns
    -------
    
    lagrange_basis_poly   : list
                            A list of size `x.shape[0]` containing the
                            analytical form of the Lagrange basis polynomials
                            in numpy.poly1d form. This list is used in
                            integrate() function which requires the analytical
                            form of the integrand.
    lagrange_basis_coeffs : numpy.ndarray
                            A :math: `N \\times N` matrix containing the
                            coefficients of the Lagrange basis polynomials such
                            that :math:`i^{th}` lagrange polynomial will be the
                            :math:`i^{th}` row of the matrix.
    **Examples**
    
    lagrange_polynomials(4)[0] gives the lagrange polynomials obtained using
    4 LGL points in poly1d form
    lagrange_polynomials(4)[0][2] is :math: `L_2(\\xi)`
    lagrange_polynomials(4)[1] gives the coefficients of the above mentioned
    lagrange basis polynomials in a 2D array.
    lagrange_polynomials(4)[1][2] gives the coefficients of :math:`L_2(\\xi)`
    in the form [a^2_3, a^2_2, a^2_1, a^2_0]
    '''
    X = np.array(x)
    lagrange_basis_poly   = []
    lagrange_basis_coeffs = np.zeros([X.shape[0], X.shape[0]])
    
    for j in np.arange(X.shape[0]):
        lagrange_basis_j = np.poly1d([1])
        
        for m in np.arange(X.shape[0]):
            if m != j:
                lagrange_basis_j *= np.poly1d([1, -X[m]]) \
                                    / (X[j] - X[m])
        lagrange_basis_poly.append(lagrange_basis_j)
        lagrange_basis_coeffs[j] = lagrange_basis_j.c
    
    return lagrange_basis_poly, lagrange_basis_coeffs


def lagrange_function_value(lagrange_coeff_array, xi_LGL):
    '''
    Funtion to calculate the value of lagrange basis functions over LGL
    nodes.

    Parameters
    ----------

    lagrange_coeff_array : arrayfire.Array[N_LGL N_LGL 1 1]
                           Contains the coefficients of the
                           Lagrange basis polynomials

    Returns
    -------

    L_i : arrayfire.Array [N 1 1 1]
          The value of lagrange basis functions calculated over the LGL
          nodes.

    **Examples**

    lagrange_function_value(4) gives the value of the four
    Lagrange basis functions evaluated over 4 LGL points
    arranged in a 2D array where Lagrange polynomials
    evaluated at the same LGL point are in the same column.

    Also the value lagrange basis functions at LGL points has the property,

    L_i(xi_k) = 0 for i != k
              = 1 for i  = k

    It follows then that lagrange_function_value returns an identity matrix.
    '''
    xi_tile    = af.transpose(af.tile(xi_LGL, 1, params.N_LGL))
    power      = af.flip(af.range(params.N_LGL))
    power_tile = af.tile(power, 1, params.N_LGL)
    xi_pow     = af.arith.pow(xi_tile, power_tile)
    index      = af.range(params.N_LGL)
    L_i        = af.blas.matmul(lagrange_coeff_array, xi_pow)
    
    return L_i



def integrate(integrand_coeffs, gv):
    '''
    Performs integration according to the given quadrature method
    by taking in the coefficients of the polynomial and the number of
    quadrature points.
    The number of quadrature points and the quadrature scheme are set
    in params.py module.

    Parameters
    ----------

    integrand_coeffs : arrayfire.Array [M N 1 1]
                       The coefficients of M number of polynomials of order N
                       arranged in a 2D array.
    Returns
    -------

    Integral : arrayfire.Array [M 1 1 1]
               The value of the definite integration performed using the
               specified quadrature method for M polynomials.
    '''
    integrand      = integrand_coeffs

    if (params.scheme == 'gauss_quadrature'):

        gaussian_nodes = gv.gauss_points
        Gauss_weights  = gv.gauss_weights

        nodes_tile   = af.transpose(af.tile(gaussian_nodes,
                                            1, integrand.shape[1]))
        power        = af.flip(af.range(integrand.shape[1]))
        nodes_power  = af.broadcast(utils.power,
                                    nodes_tile, power)
        weights_tile = af.transpose(af.tile(Gauss_weights, 1,
                                            integrand.shape[1]))
        nodes_weight = nodes_power * weights_tile

        value_at_gauss_nodes = af.matmul(integrand, nodes_weight)
        integral             = af.sum(value_at_gauss_nodes, 1)
 
    if (params.scheme == 'lobatto_quadrature'):

        lobatto_nodes   = gv.lobatto_quadrature_nodes
        Lobatto_weights = gv.lobatto_weights_quadrature

        nodes_tile   = af.transpose(af.tile(lobatto_nodes, 1,
                                            integrand.shape[1]))
        power        = af.flip(af.range(integrand.shape[1]))
        nodes_power  = af.broadcast(utils.power, nodes_tile, power)
        weights_tile = af.transpose(af.tile(Lobatto_weights, 1,
                                            integrand.shape[1]))
        nodes_weight = nodes_power * weights_tile


        value_at_lobatto_nodes = af.matmul(integrand, nodes_weight)
        integral               = af.sum(value_at_lobatto_nodes, 1)

    return integral

def lagrange_interpolation_u(u, gv):
    '''
    Calculates the coefficients of the Lagrange interpolation using
    the value of u at the mapped LGL points in the domain.
    The interpolation using the Lagrange basis polynomials is given by
    :math:`L_i(\\xi) u_i(\\xi)`
    Where L_i are the Lagrange basis polynomials and u_i is the value
    of u at the LGL points.
    
    Parameters
    ----------
    u : arrayfire.Array [N_LGL N_Elements 1 1]
        The value of u at the mapped LGL points.
        
    Returns
    -------
    lagrange_interpolated_coeffs : arrayfire.Array[1 N_LGL N_Elements 1]
                                   The coefficients of the polynomials obtained
                                   by Lagrange interpolation. Each polynomial
                                   is of order N_LGL - 1.
    '''
    lagrange_coeffs_tile = af.tile(gv.lagrange_coeffs, 1, 1,\
                                               params.N_Elements)
    reordered_u          = af.reorder(u, 0, 2, 1)

    lagrange_interpolated_coeffs = af.sum(af.broadcast(utils.multiply,\
                                             reordered_u, lagrange_coeffs_tile), 0)

    return lagrange_interpolated_coeffs


def L1_norm(u):
    '''
    A function to calculate the L1 norm of error using
    the polynomial obtained using Lagrange interpolation
    Parameters
    ----------
    u : arrayfire.Array [N_LGL N_Elements 1 1]
        Difference between analytical and numerical u at the mapped LGL points.

    Returns
    -------
    L1_norm : float64
              The L1 norm of error.
    '''
    interpolated_coeffs = af.reorder(lagrange_interpolation_u(u),
                                     2, 1, 0)

    L1_norm = af.sum(integrate(interpolated_coeffs))
    
    return L1_norm



def Li_basis_value(L_basis, i, xi):
    '''
    Finds the value of the :math:`i^{th}` Lagrange basis polynomial
    at the given :math:`\\xi` coordinates.

    Parameters
    ----------
    L_basis : af.Array [N_LGL N_LGL 1 1]
              Lagrange basis polynomial coefficient array

    i       : af.Array [N 1 1 1]
              Index of the Lagrange basis polynomials
              to be evaluated.

    xi      : af.Array [N 1 1 1]
              :math:`\\xi` coordinates at which the :math:`i^{th}` Lagrange
              basis polynomial is to be evaluated.

    Returns
    -------
    af.Array [i.shape[0] xi.shape[0] 1 1]
        Evaluated :math:`i^{th}` lagrange basis polynomials at given
        :math:`\\xi` coordinates
    '''

    return utils.polyval_1d(L_basis[i], xi)
