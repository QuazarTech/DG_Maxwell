#! /usr/bin/env python3

import numpy as np
import arrayfire as af
from scipy import special as sp

from utils import utils
from app import params

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
                          
    Reference
    ---------
    `https://goo.gl/KdG2Sv`
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
    Reference
    ---------
    Gauss-Lobatto weights Wikipedia link-
    https://en.wikipedia.org/wiki/
    Gaussian_quadrature#Gauss.E2.80.93Lobatto_rules


    Examples
    --------
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

    Reference
    ---------
    A Wikipedia article about the Gauss-Legendre quadrature
    `https://goo.gl/9gqLpe`
    '''
    legendre = sp.legendre(n)
    gauss_nodes = legendre.r
    gauss_nodes.sort()
    
    return gauss_nodes


def gaussian_weights(N):
    '''
    Returns the gaussian weights :math:`w_i` for :math: `N` Gaussian Nodes
    at index :math: `i`. They are given by

    :math: `w_i = \\frac{2}{(1 - x_i^2) P'n(x_i)}`

    Where :math:`x_i` are the Gaussian nodes and :math: `P_{n}(\\xi)` 
    are the Legendre polynomials.
    

    Parameters
    ----------
    N : int
        Number of Gaussian nodes for which the weight is t be calculated.
            
   
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
    :math::
    `L_i = \\prod_{m = 0, m \\notin i}^{N - 1}\\frac{(x - x_m)}{(x_i - x_m)}`

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
                            Integrate() function which requires the analytical
                            form of the integrand.

    lagrange_basis_coeffs : numpy.ndarray
                            A :math: `N \\times N` matrix containing the
                            coefficients of the Lagrange basis polynomials such
                            that :math:`i^{th}` lagrange polynomial will be the
                            :math:`i^{th}` row of the matrix.
    Examples
    --------
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

    Examples
    --------
    lagrange_function_value(4) gives the value of the four 
    Lagrange basis functions evaluated over 4 LGL points
    arranged in a 2D array where Lagrange polynomials
    evaluated at the same LGL point are in the same column.
    
    Also the value lagrange basis functions at LGL points has the property,
    
    L_i(xi_k) = 0 for i != k
              = 1 for i = k
    
    It follows then that lagrange_function_value returns an identity matrix.
    
    '''
    xi_tile    = af.transpose(af.tile(params.xi_LGL, 1, params.N_LGL))
    power      = af.flip(af.range(params.N_LGL))
    power_tile = af.tile(power, 1, params.N_LGL)
    xi_pow     = af.arith.pow(xi_tile, power_tile)
    index      = af.range(params.N_LGL)
    L_i        = af.blas.matmul(lagrange_coeff_array[index], xi_pow)
    
    return L_i


def product_lagrange_poly(x):
    '''
    Used to obtain the coefficients of the product of Lagrange polynomials.

    A matrix involves integrals of the product of the Lagrange polynomials.
    The Integrate() function requires the coefficients of the integrand to
    compute the integral.

    This function takes the poly1d form of the Lagrange basis polynomials,
    multiplies them and stores the coefficients in a 2D array.

    Parameters
    ----------
    x : arrayfire.Array[N_LGL 1 1 1]
        Contains N_LGL Gauss-Lobatto nodes.

    Returns
    -------
    lagrange_product_coeffs : arrayfire.Array [N_LGL**2 N_LGL*2-1 1 1]
                              Contains the coefficients of the product of the
                              Lagrange polynomials.

    Examples
    --------
    product_lagrange_poly(xi_LGL)[0] gives the coefficients of the product
    `L_0(\\xi) * L_0(\\xi)`.


    product_lagrange_poly(xi_LGL)[1] gives the coefficients of the product
    `L_0(\\xi) * L_1(\\xi)`.
                              
    '''
    poly1d_list             = lagrange_polynomials(params.xi_LGL)[0] 
    lagrange_product_coeffs = np.zeros([params.N_LGL ** 2, params.N_LGL * 2 - 1])

    for i in range (params.N_LGL):
        for j in range (params.N_LGL):
            lagrange_product_coeffs[params.N_LGL * i + j] = ((poly1d_list[i] * poly1d_list[j]).c)

    lagrange_product_coeffs = af.np_to_af_array(lagrange_product_coeffs)

    return lagrange_product_coeffs


def Integrate(integrand_coeffs):
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

    if (params.scheme == 'gauss_quadrature'):
        
        integrand      = integrand_coeffs
        gaussian_nodes = params.gauss_points
        Gauss_weights  = params.gauss_weights
        
        nodes_tile   = af.transpose(af.tile(gaussian_nodes, 1, integrand.shape[1]))
        power        = af.flip(af.range(integrand.shape[1]))
        nodes_power  = af.broadcast(utils.power, nodes_tile, power)
        weights_tile = af.transpose(af.tile(Gauss_weights, 1, integrand.shape[1]))
        nodes_weight = nodes_power * weights_tile

        value_at_gauss_nodes = af.matmul(integrand, nodes_weight)
        Integral             = af.sum(value_at_gauss_nodes, 1)
 
    if (params.scheme == 'lobatto_quadrature'):

        integrand       = integrand_coeffs
        lobatto_nodes   = params.lobatto_quadrature_nodes
        Lobatto_weights = params.lobatto_weights_quadrature

        nodes_tile   = af.transpose(af.tile(lobatto_nodes, 1, integrand.shape[1]))
        power        = af.flip(af.range(integrand.shape[1]))
        nodes_power  = af.broadcast(utils.power, nodes_tile, power)
        weights_tile = af.transpose(af.tile(Lobatto_weights, 1, integrand.shape[1]))
        nodes_weight = nodes_power * weights_tile

        
        value_at_lobatto_nodes = af.matmul(integrand, nodes_weight)
        Integral               = af.sum(value_at_lobatto_nodes, 1)

    
    return Integral


def wave_equation_lagrange_basis_single_element(u, element_no):
    '''
    Calculates the function which describes the amplitude of the wave in
    a particular element.

    Using the value of the amplitude at the LGL points, A function which
    describes this behaviour is obtained by expressing it as a linear
    combination of the Lagrange basis polynomials.

    :math: ` f(x) = '\\sigma_i a_i L_i(\\xi)`

    Where the coefficients a_i are the value of the function at the
    LGL points.

    Parameters
    ----------
    u          : arrayfire.Array [N_LGL N_Elements 1 1]
                 The amplitude of the wave at the LGL points for a
                 single element.

    element_no : int
                 The element for which the analytical form of the wave equation
                 is required.

    Returns
    -------
    wave_equation_element : numpy.poly1d
                            The analytical form of the function which describes
                            the amplitude locally.
    '''
    amplitude_at_element_LGL   = u[:, element_no]
    lagrange_basis_polynomials = params.lagrange_poly1d_list
    
    wave_equation_element = np.poly1d([0])

    for i in range(0, params.N_LGL):
        wave_equation_element += af.sum(amplitude_at_element_LGL[i])\
                                        * lagrange_basis_polynomials[i]

    return wave_equation_element

def wave_equation_lagrange(u):
    '''
    Calculates the local wave equation in the Lagrange basis space
    for all elements using wave_equation_lagrange_basis_single_element function.

    Parameters
    ----------
    u : arrayfire.Array [N_LGL N_Elements 1 1]
        Contains the amplitude of the wave at the LGL points for all elements.

    Returns
    -------
    wave_equation_lagrange_basis : list [N_Elements]
                                   Contains the local approximation of the wave
                                   function in the form of a list
    '''
    wave_equation_lagrange_basis = []

    for i in range(0, params.N_Elements):
        element_wave_equation = wave_equation_lagrange_basis_single_element(u, i)

        wave_equation_lagrange_basis.append(element_wave_equation)  

    return wave_equation_lagrange_basis

def differential_lagrange_poly1d():
    '''
    Calculates the differential of the analytical form of the Lagrange basis
    polynomials.

    Returns
    -------
    diff_lagrange_poly1d : list [N_LGL]
                           Contains the differential of the Lagrange basis
                           polynomials in numpy.poly1d form.
    '''
    diff_lagrange_poly1d = []

    for i in range (0, params.N_LGL):
        test_diff = np.poly1d.deriv(params.lagrange_poly1d_list[i])
        diff_lagrange_poly1d.append(test_diff)
    
    return diff_lagrange_poly1d 


def max_amplitude(u_t_n):
    '''
    '''
    element_function   = wave_equation_lagrange(u_t_n)
    linspace_nos       = utils.linspace(-1, 1, 100)
    linspace_tile      = af.transpose(af.tile(linspace_nos, 1, params.N_Elements))
    random_nos         = linspace_tile

    amp = 0
    for i in range(0, params.N_Elements):
        element_max = max(max(abs(element_function[i](random_nos[i, :]))))
        if amp < element_max:
            amp = element_max

    return amp
