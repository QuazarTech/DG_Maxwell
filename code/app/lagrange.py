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


def lobatto_weights(n, x):
    '''
    Calculates the weight function for an index :math:`n`
    and points :math: `x`.
    
    :math::
        `w_{n} = \\frac{2 P(x)^2}{n (n - 1)}`,
        Where P(x) is $ (n - 1)^th $ index.
    
    Parameters
    ----------
    n : int
        Index for which lobatto weight function
    
    x : arrayfire.Array [N 1 1 1]
        Points where weight function is to be calculated.
    
    
    Returns
    -------
    gauss_lobatto_weights : numpy.ndarray [N 1 1 1]
                            Lobatto weight for the given :math: `x`
                            points and index `i`.
    Reference
    ---------
    Gauss-Lobatto weights Wikipedia link-
    `https://goo.gl/o7WE4K`
    '''
    P = sp.legendre(n - 1)

    gauss_lobatto_weights = (2 / (n * (n - 1)) / (P(x))**2)

    return gauss_lobatto_weights

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

def gaussian_weights(N, i):
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
            
    i : int
        Index for which the Gaussian weight is required.
    
    Returns
    -------
    gaussian_weight : double 
                      The gaussian weight.
    
    '''
    
    gaussian_nodes = gauss_nodes(N)
    gaussian_weight  = 2 / ((1 - (gaussian_nodes[i]) ** 2) *\
                       (np.polyder(sp.legendre(N))(gaussian_nodes[i])) ** 2)
    
    
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
    Calculates the product of Lagrange basis polynomials in 'np.poly1d' in a 
    2D array. This analytical form of the product of the Lagrange basis is used
    in the calculation of A matrix using the Integrate() function.
    
    Parameters
    ----------
    x : arrayfire.Array[N_LGL 1 1 1]
        Contains N_LGL Gauss-Lobatto nodes.

    Returns
    -------
    poly1d_product_list : list [N_LGL ** 2]
                          Contains the poly1d form of the product of the Lagrange
                          basis polynomials.
    '''
    poly1d_list = lagrange_polynomials(params.xi_LGL)[0] 
    poly1d_product_coeffs = np.zeros([params.N_LGL ** 2, params.N_LGL * 2 - 1])

    for i in range (params.N_LGL):
        for j in range (params.N_LGL):
            poly1d_product_coeffs[params.N_LGL * i + j] = ((poly1d_list[i] * poly1d_list[j]).c)

    poly1d_product_coeffs = af.np_to_af_array(poly1d_product_coeffs)

    return poly1d_product_coeffs



def Integrate(integrand_coeffs, N_quad, scheme):
    '''
    Performs integration according to the given quadrature method
    by taking in the coefficients of the polynomial and the number of
    quadrature points.
    
    Parameters
    ----------
    integrand_coeffs : arrayfire.Array [M N 1 1]
                       The coefficients of M number of polynomials of order N
                       arranged in a 2D array.

    N_quad           : int
                       The number of quadrature points to be used for Integration
                       using either Gaussian or Lobatto quadrature.    

    scheme           : string
                       Specifies the method of integration to be used. Can take values
                       'gauss_quadrature' and 'lobatto_quadrature'.

    Returns
    -------
    Integral : arrayfire.Array [M 1 1 1]
               The value of the definite integration performed using the
               specified quadrature method for M polynomials.
    '''
    if (scheme == 'gauss_quadrature'):
        integrand  = (integrand_coeffs)
        gaussian_nodes = af.np_to_af_array(gauss_nodes(N_quad))
        gauss_weights  = af.np_to_af_array(np.zeros([N_quad]))
        
        
        for i in range(0, N_quad):
            gauss_weights[i] = gaussian_weights(N_quad, i)

         
        nodes_tile  = af.transpose(af.tile(gaussian_nodes, 1, integrand.shape[1]))
        power       = af.flip(af.range(integrand.shape[1]))
        nodes_power = af.broadcast(utils.power, nodes_tile, power)
        weights_tile = af.transpose(af.tile(gauss_weights, 1, integrand.shape[1]))
        nodes_weight = nodes_power * weights_tile

        
        value_at_gauss_nodes = af.matmul(integrand, nodes_weight)
        Integral = af.sum(value_at_gauss_nodes, 1)
        
    if (scheme == 'lobatto_quadrature'):

        integrand  = (integrand_coeffs)
        lobatto_nodes = (LGL_points(N_quad))
        Lobatto_weights  = af.np_to_af_array(lobatto_weights(N_quad, lobatto_nodes))
        
       
        nodes_tile  = af.transpose(af.tile(lobatto_nodes, 1, integrand.shape[1]))
        power       = af.flip(af.range(integrand.shape[1]))
        nodes_power = af.broadcast(utils.power, nodes_tile, power)
        weights_tile = af.transpose(af.tile(Lobatto_weights, 1, integrand.shape[1]))
        nodes_weight = nodes_power * weights_tile

        
        value_at_lobatto_nodes = af.matmul(integrand, nodes_weight)
        Integral = af.sum(value_at_lobatto_nodes, 1)

    
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
    
    return  diff_lagrange_poly1d 
