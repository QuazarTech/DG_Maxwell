#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy import special as sp
import arrayfire as af
af.set_backend('cpu')

from dg_maxwell import utils
from dg_maxwell import params


LGL_list = [ \
[-1.0,1.0],                                                               \
[-1.0,0.0,1.0],                                                           \
[-1.0,-0.4472135955,0.4472135955,1.0],                                    \
[-1.0,-0.654653670708,0.0,0.654653670708,1.0],                            \
[-1.0,-0.765055323929,-0.285231516481,0.285231516481,0.765055323929,1.0], \
[-1.0,-0.830223896279,-0.468848793471,0.0,0.468848793471,0.830223896279,  \
1.0],                                                                     \
[-1.0,-0.87174014851,-0.591700181433,-0.209299217902,0.209299217902,      \
0.591700181433,0.87174014851,1.0],                                        \
[-1.0,-0.899757995411,-0.677186279511,-0.363117463826,0.0,0.363117463826, \
0.677186279511,0.899757995411,1.0],                                       \
[-1.0,-0.919533908167,-0.738773865105,-0.47792494981,-0.165278957666,     \
0.165278957666,0.47792494981,0.738773865106,0.919533908166,1.0],          \
[-1.0,-0.934001430408,-0.784483473663,-0.565235326996,-0.295758135587,    \
0.0,0.295758135587,0.565235326996,0.784483473663,0.934001430408,1.0],     \
[-1.0,-0.944899272223,-0.819279321644,-0.632876153032,-0.399530940965,    \
-0.136552932855,0.136552932855,0.399530940965,0.632876153032,             \
0.819279321644,0.944899272223,1.0],                                       \
[-1.0,-0.953309846642,-0.846347564652,-0.686188469082,-0.482909821091,    \
-0.249286930106,0.0,0.249286930106,0.482909821091,0.686188469082,         \
0.846347564652,0.953309846642,1.0],                                       \
[-0.999999999996,-0.959935045274,-0.867801053826,-0.728868599093,         \
-0.550639402928,-0.342724013343,-0.116331868884,0.116331868884,           \
0.342724013343,0.550639402929,0.728868599091,0.86780105383,               \
0.959935045267,1.0],                                                      \
[-0.999999999996,-0.965245926511,-0.885082044219,-0.763519689953,         \
-0.60625320547,-0.420638054714,-0.215353955364,0.0,0.215353955364,        \
0.420638054714,0.60625320547,0.763519689952,0.885082044223,               \
0.965245926503,1.0],                                                      \
[-0.999999999984,-0.9695680463,-0.899200533072,-0.792008291871,           \
-0.65238870288,-0.486059421887,-0.299830468901,-0.101326273522,           \
0.101326273522,0.299830468901,0.486059421887,0.652388702882,              \
0.792008291863,0.899200533092,0.969568046272,0.999999999999]]



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
    lgl_points = np.array(LGL_list[N - 2])
    lgl_points = af.np_to_af_array(lgl_points)

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
    lagrange_basis_coeffs = af.np_to_af_array(np.zeros([X.shape[0], X.shape[0]]))
    
    for j in np.arange(X.shape[0]):
        lagrange_basis_k = af.np_to_af_array(np.array([1.]))
        
        for m in np.arange(X.shape[0]):
            if m != j:
                lagrange_basis_k = af.convolve1(lagrange_basis_k,\
                        af.np_to_af_array(np.array([1, -X[m]])/ (X[j] - X[m])), conv_mode=af.CONV_MODE.EXPAND)
        lagrange_basis_coeffs[j] = af.transpose(lagrange_basis_k)
    
    return lagrange_basis_coeffs


def lagrange_function_value(lagrange_coeff_array):
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
    xi_tile    = af.transpose(af.tile(params.xi_LGL, 1, params.N_LGL))
    power      = af.flip(af.range(params.N_LGL))
    power_tile = af.tile(power, 1, params.N_LGL)
    xi_pow     = af.arith.pow(xi_tile, power_tile)
    index      = af.range(params.N_LGL)
    L_i        = af.blas.matmul(lagrange_coeff_array, xi_pow)
    
    return L_i



def integrate(integrand_coeffs):
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
        #print('gauss_quad')

        gaussian_nodes = params.gauss_points
        Gauss_weights  = params.gauss_weights

        nodes_tile   = af.transpose(af.tile(gaussian_nodes, 1, integrand.shape[1]))
        power        = af.flip(af.range(integrand.shape[1]))
        nodes_power  = af.broadcast(utils.power, nodes_tile, power)
        weights_tile = af.transpose(af.tile(Gauss_weights, 1, integrand.shape[1]))
        nodes_weight = nodes_power * weights_tile

        value_at_gauss_nodes = af.matmul(integrand, nodes_weight)
        integral             = af.sum(value_at_gauss_nodes, 1)
 
    if (params.scheme == 'lobatto_quadrature'):
        #print('lob_quad')

        lobatto_nodes   = params.lobatto_quadrature_nodes
        Lobatto_weights = params.lobatto_weights_quadrature

        nodes_tile   = af.transpose(af.tile(lobatto_nodes, 1, integrand.shape[1]))
        power        = af.flip(af.range(integrand.shape[1]))
        nodes_power  = af.broadcast(utils.power, nodes_tile, power)
        weights_tile = af.transpose(af.tile(Lobatto_weights, 1, integrand.shape[1]))
        nodes_weight = nodes_power * weights_tile


        value_at_lobatto_nodes = af.matmul(integrand, nodes_weight)
        integral               = af.sum(value_at_lobatto_nodes, 1)


    return integral


def integrate_2D(f_coeffs, g_coeffs):
    '''

    Parameters
    ----------
    f_coeffs : arrayfire.Array [N M 1 1]
               The coeffeicients of N polynomials of order M - 1 and
               variable (say x)

    g_coeffs : arrayfire.Array [N M 1 1]
               The coeffecients of N polynomials of order M - 1 and
               a different variable (say y)

    Returns
    -------
    Integral : arrayfire.Array [ N 1 1 1]
               The integral of the product of the two polynomials
               over the two variables from -1 to 1.

    '''

    if (params.scheme == 'gauss_quadrature'):
        #print('gauss_quad')

        gaussian_nodes = params.gauss_points
        Gauss_weights  = params.gauss_weights

        nodes_tile   = af.transpose(af.tile(gaussian_nodes, 1, f_coeffs.shape[1]))
        power        = af.flip(af.range(f_coeffs.shape[1]))
        nodes_power  = af.broadcast(utils.power, nodes_tile, power)
        weights_tile = af.transpose(af.tile(Gauss_weights, 1, f_coeffs.shape[1]))
        nodes_weight = nodes_power * weights_tile

        value_at_gauss_nodes_f = af.matmul(f_coeffs, nodes_weight)

        nodes_tile   = af.transpose(af.tile(gaussian_nodes, 1, g_coeffs.shape[1]))
        power        = af.flip(af.range(g_coeffs.shape[1]))
        nodes_power  = af.broadcast(utils.power, nodes_tile, power)
        weights_tile = af.transpose(af.tile(Gauss_weights, 1, g_coeffs.shape[1]))
        nodes_weight = nodes_power * weights_tile

        value_gauss_nodes_g = af.matmul(g_coeffs, nodes_weight)

        value_gauss_nodes_f = af.reorder(value_at_gauss_nodes_f, 0, 2, 1)

        integral = af.broadcast(utils.multiply, value_gauss_nodes_f, value_gauss_nodes_g)
        integral = af.sum(integral, 2)
        integral = af.sum(integral,1)


    return integral



def lagrange_interpolation_u(u):
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
    lagrange_coeffs_tile = af.tile(params.lagrange_coeffs, 1, 1,\
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
    interpolated_coeffs = af.reorder(lagrange_interpolation_u(\
                                           u), 2, 1, 0)

    L1_norm = af.sum(integrate(interpolated_coeffs))

    return L1_norm
