#! /usr/bin/env python3
# -*- coding: utf-8 -*-
    
import os

import arrayfire as af
af.set_backend('opencl')
af.set_device(1)

import numpy as np
from matplotlib import pyplot as plt
from tqdm import trange
import h5py
from scipy import integrate

from dg_maxwell import params
from dg_maxwell import lagrange
from dg_maxwell import utils
from dg_maxwell import isoparam

plt.rcParams['figure.figsize'  ] = 9.6, 6.
plt.rcParams['figure.dpi'      ] = 100
plt.rcParams['image.cmap'      ] = 'jet'
plt.rcParams['lines.linewidth' ] = 1.5
plt.rcParams['font.family'     ] = 'serif'
plt.rcParams['font.weight'     ] = 'bold'
plt.rcParams['font.size'       ] = 20
plt.rcParams['font.sans-serif' ] = 'serif'
plt.rcParams['text.usetex'     ] = True
plt.rcParams['axes.linewidth'  ] = 1.5
plt.rcParams['axes.titlesize'  ] = 'medium'
plt.rcParams['axes.labelsize'  ] = 'medium'
plt.rcParams['xtick.major.size'] = 8
plt.rcParams['xtick.minor.size'] = 4
plt.rcParams['xtick.major.pad' ] = 8
plt.rcParams['xtick.minor.pad' ] = 8
plt.rcParams['xtick.color'     ] = 'k'
plt.rcParams['xtick.labelsize' ] = 'medium'
plt.rcParams['xtick.direction' ] = 'in'
plt.rcParams['ytick.major.size'] = 8
plt.rcParams['ytick.minor.size'] = 4
plt.rcParams['ytick.major.pad' ] = 8
plt.rcParams['ytick.minor.pad' ] = 8
plt.rcParams['ytick.color'     ] = 'k'
plt.rcParams['ytick.labelsize' ] = 'medium'
plt.rcParams['ytick.direction' ] = 'in'


def mapping_xi_to_x(x_nodes, xi):
    '''
    Maps points in :math: `\\xi` space to :math:`x` space using the formula
    :math:  `x = \\frac{1 - \\xi}{2} x_0 + \\frac{1 + \\xi}{2} x_1`
    
    Parameters
    ----------
    
    x_nodes : arrayfire.Array [2 1 1 1]
              Element nodes.
    
    xi      : arrayfire.Array [N 1 1 1]
              Value of :math: `\\xi`coordinate for which the corresponding
              :math: `x` coordinate is to be found.
    
    Returns
    -------

    x : arrayfire.Array
        :math: `x` value in the element corresponding to :math:`\\xi`.
    '''

    N_0 = (1 - xi) / 2
    N_1 = (1 + xi) / 2
    
    N0_x0 = af.broadcast(utils.multiply, N_0, x_nodes[0])
    N1_x1 = af.broadcast(utils.multiply, N_1, x_nodes[1])
    
    x = N0_x0 + N1_x1
    
    return x


def dx_dxi_numerical(x_nodes, xi):
    '''

    Differential :math: `\\frac{dx}{d \\xi}` calculated by central
    differential method about xi using the mapping_xi_to_x function.
    
    Parameters
    ----------
    
    x_nodes : arrayfire.Array [N_Elements 1 1 1]
              Contains the nodes of elements.
    
    xi      : arrayfire.Array [N_LGL 1 1 1]
              Values of :math: `\\xi`
    
    Returns
    -------

    dx_dxi : arrayfire.Array [N_Elements 1 1 1]
             :math:`\\frac{dx}{d \\xi}`. 
    '''

    dxi = 1e-7
    x2  = mapping_xi_to_x(x_nodes, xi + dxi)
    x1  = mapping_xi_to_x(x_nodes, xi - dxi)
    
    dx_dxi = (x2 - x1) / (2 * dxi)
    
    return dx_dxi


def dx_dxi_analytical(x_nodes, xi):
    '''

    The analytical result for :math:`\\frac{dx}{d \\xi}` for a 1D element is
    :math: `\\frac{x_1 - x_0}{2}`
    
    Parameters
    ----------

    x_nodes : arrayfire.Array [2 N_Elements 1 1]
              Contains the nodes of elements.
 
    xi      : arrayfire.Array [N_LGL 1 1 1]
              Values of :math: `\\xi`.

    Returns
    -------

    analytical_dx_dxi : arrayfire.Array [N_Elements 1 1 1]
                        The analytical solution of :math:
                        `\\frac{dx}{d\\xi}` for an element.
    
    '''
    analytical_dx_dxi = (x_nodes[1] - x_nodes[0]) / 2
    
    return analytical_dx_dxi


def A_matrix():
    '''

    Calculates A matrix whose elements :math:`A_{p i}` are given by
    :math: `A_{p i} &= \\int^1_{-1} L_p(\\xi)L_i(\\xi) \\frac{dx}{d\\xi}`

    The integrals are computed using the integrate() function.
    Since elements are taken to be of equal size, :math: `\\frac {dx}{dxi}
    is same everywhere
    
    Returns
    -------

    A_matrix : arrayfire.Array [N_LGL N_LGL 1 1]
               The value of integral of product of lagrange basis functions
               obtained by LGL points, using the integrate() function

    '''
    # Coefficients of Lagrange basis polynomials.
    lagrange_coeffs = params.lagrange_coeffs
    lagrange_coeffs = af.reorder(lagrange_coeffs, 1, 2, 0)

    # Coefficients of product of Lagrange basis polynomials.
    lag_prod_coeffs = af.convolve1(lagrange_coeffs,\
                                   af.reorder(lagrange_coeffs, 0, 2, 1),\
                                   conv_mode=af.CONV_MODE.EXPAND)
    lag_prod_coeffs = af.reorder(lag_prod_coeffs, 1, 2, 0)
    lag_prod_coeffs = af.moddims(lag_prod_coeffs, params.N_LGL ** 2, 2 * params.N_LGL - 1)


    dx_dxi   = params.dx_dxi 
    A_matrix = dx_dxi * af.moddims(lagrange.integrate(lag_prod_coeffs),\
                                             params.N_LGL, params.N_LGL)
    
    return A_matrix


def flux_x(u):
    '''

    A function which returns the value of flux for a given wave function u.
    :math:`f(u) = c u^k`
    
    Parameters
    ----------

    u : list [N_Elements]
        The analytical form of the wave equation for each element arranged in
        a list of numpy.poly1d polynomials.

    Returns
    -------

    flux : list [N_Elements] 
           The analytical value of the flux for each element arranged in a list
           of numpy.poly1d polynomials.

    '''
    #flux = params.c * u
    flux = u.copy()
    
    flux[:, :, 0] = -u[:, :, 1]
    flux[:, :, 1] = -u[:, :, 0]
    
    return flux


def flux_maxwell(cons_var):
    '''
    Fluxes for the :math:`(E_z, B_y)` mode in :math:`1D` Maxwell's equations.
    
    Parameters
    ----------
    
    cons_var : af.Array [N_Elements 2]
    '''
    
    E_z = cons_var[0]
    B_y = cons_var[1]
    
    return (-B_y, E_z)


def volume_integral_flux(u_n):
    '''

    Calculates the volume integral of flux in the wave equation.

    :math:`\\int_{-1}^1 f(u) \\frac{d L_p}{d\\xi} d\\xi`

    This will give N values of flux integral as p varies from 0 to N - 1.
    
    This integral is carried out using the analytical form of the integrand
    obtained as a linear combination of Lagrange basis polynomials.

    This integrand is the used in the integrate() function.

    Calculation of volume integral flux using N_LGL Lobatto quadrature points
    can be vectorized and is much faster.
    
    Parameters
    ----------

    u : arrayfire.Array [N_LGL N_Elements 1 1]
        Amplitude of the wave at the mapped LGL nodes of each element.
            
    Returns
    -------

    flux_integral : arrayfire.Array [N_LGL N_Elements 1 1]
                    Value of the volume integral flux. It contains the integral
                    of all N_LGL * N_Element integrands.

    '''
    # The coefficients of dLp / d\xi
    diff_lag_coeff  = params.dl_dxi_coeffs

    lobatto_nodes   = params.lobatto_quadrature_nodes
    Lobatto_weights = params.lobatto_weights_quadrature

    nodes_tile   = af.transpose(af.tile(lobatto_nodes, 1, diff_lag_coeff.shape[1]))
    power        = af.flip(af.range(diff_lag_coeff.shape[1]))
    power_tile   = af.tile(power, 1, params.N_quad)
    nodes_power  = nodes_tile ** power_tile
    weights_tile = af.transpose(af.tile(Lobatto_weights, 1, diff_lag_coeff.shape[1]))
    nodes_weight = nodes_power * weights_tile

    dLp_dxi      = af.matmul(diff_lag_coeff, nodes_weight)


    # The first option to calculate the volume integral term, directly uses
    # the Lobatto quadrature instead of using the integrate() function by
    # passing the coefficients of the Lagrange interpolated polynomial.
    if(params.volume_integral_scheme == 'lobatto_quadrature'\
        and params.N_quad == params.N_LGL):

        # Flux using u_n, reordered to 1 X N_LGL X N_Elements array.
        F_u_n                  = af.reorder(flux_x(u_n), 2, 0, 1)

        # Multiplying with dLp / d\xi
        integral_expansion     = af.broadcast(utils.multiply,\
                                 dLp_dxi, F_u_n)

        # Using the quadrature rule.
        flux_integral = af.sum(integral_expansion, 1)
        flux_integral = af.reorder(flux_integral, 0, 2, 1)

    # Using the integrate() function to calculate the volume integral term
    # by passing the Lagrange interpolated polynomial.
    else:
        #print('option3')
        analytical_flux_coeffs = flux_x(lagrange.\
                                        lagrange_interpolation_u(u_n))

        analytical_flux_coeffs = af.reorder(analytical_flux_coeffs, 1, 0, 2)

        dl_dxi_coefficients    = af.reorder(params.dl_dxi_coeffs, 1, 0)

        # The product of polynomials is calculated using af.convolve1
        volume_int_coeffs = af.convolve1(dl_dxi_coefficients,\
                                         analytical_flux_coeffs,\
                                         conv_mode=af.CONV_MODE.EXPAND)
        volume_int_coeffs = af.reorder(volume_int_coeffs, 1, 2, 0)
        volume_int_coeffs = af.moddims(volume_int_coeffs,\
                                       params.N_LGL * params.N_Elements,\
                                       2 * params.N_LGL - 2)


        flux_integral = lagrange.integrate(volume_int_coeffs)
        flux_integral = af.moddims(flux_integral, params.N_LGL, params.N_Elements)


    return flux_integral



def volume_integral_flux_multiple_u_n(u_n):
    '''

    Calculates the volume integral of flux in the wave equation.

    :math:`\\int_{-1}^1 f(u) \\frac{d L_p}{d\\xi} d\\xi`

    This will give N values of flux integral as p varies from 0 to N - 1.
    
    This integral is carried out using the analytical form of the integrand
    obtained as a linear combination of Lagrange basis polynomials.

    This integrand is the used in the integrate() function.

    Calculation of volume integral flux using N_LGL Lobatto quadrature points
    can be vectorized and is much faster.
    
    Parameters
    ----------

    u : arrayfire.Array [N_LGL N_Elements M 1]
        Amplitude of the wave at the mapped LGL nodes of each element. This
        function can computer flux for :math:`M` :math:`u`.
            
    Returns
    -------

    flux_integral : arrayfire.Array [N_LGL N_Elements M 1]
                    Value of the volume integral flux. It contains the integral
                    of all N_LGL * N_Element integrands.

    '''
    shape_u_n = utils.shape(u_n)
    
    # The coefficients of dLp / d\xi
    diff_lag_coeff  = params.dl_dxi_coeffs

    lobatto_nodes   = params.lobatto_quadrature_nodes
    Lobatto_weights = params.lobatto_weights_quadrature

    nodes_tile   = af.transpose(af.tile(lobatto_nodes, 1, diff_lag_coeff.shape[1]))
    power        = af.flip(af.range(diff_lag_coeff.shape[1]))
    power_tile   = af.tile(power, 1, params.N_quad)
    nodes_power  = nodes_tile ** power_tile
    weights_tile = af.transpose(af.tile(Lobatto_weights, 1, diff_lag_coeff.shape[1]))
    nodes_weight = nodes_power * weights_tile

    dLp_dxi      = af.matmul(diff_lag_coeff, nodes_weight)


    # The first option to calculate the volume integral term, directly uses
    # the Lobatto quadrature instead of using the integrate() function by
    # passing the coefficients of the Lagrange interpolated polynomial.
    if(params.volume_integral_scheme == 'lobatto_quadrature' \
        and params.N_quad == params.N_LGL):

        # Flux using u_n, reordered to 1 X N_LGL X N_Elements array.
        F_u_n                  = af.reorder(flux_x(u_n), 3, 0, 1, 2)
        F_u_n = af.tile(F_u_n, d0 = params.N_LGL)

        # Multiplying with dLp / d\xi
        integral_expansion     = af.broadcast(utils.multiply,
                                            dLp_dxi, F_u_n)

    #     # Using the quadrature rule.
        flux_integral = af.sum(integral_expansion, 1)

        flux_integral = af.reorder(flux_integral, 0, 2, 3, 1)

    # Using the integrate() function to calculate the volume integral term
    # by passing the Lagrange interpolated polynomial.
    else:
        #print('option3')
        analytical_flux_coeffs = af.transpose(af.moddims(u_n,
                                                        d0 = params.N_LGL,
                                                        d1 = params.N_Elements \
                                                            * shape_u_n[2]))
        
        analytical_flux_coeffs = flux_x(
            lagrange.lagrange_interpolation(analytical_flux_coeffs))
        analytical_flux_coeffs = af.transpose(
            af.moddims(af.transpose(analytical_flux_coeffs),
                       d0 = params.N_LGL, d1 = params.N_Elements,
                       d2 = shape_u_n[2]))
        
        analytical_flux_coeffs = af.reorder(analytical_flux_coeffs,
                                            d0 = 3, d1 = 1, d2 = 0, d3 = 2)
        analytical_flux_coeffs = af.tile(analytical_flux_coeffs,
                                         d0 = params.N_LGL)
        analytical_flux_coeffs = af.moddims(
            af.transpose(analytical_flux_coeffs),
            d0 = params.N_LGL,
            d1 = params.N_LGL * params.N_Elements, d2 = 1,
            d3 = shape_u_n[2])
        
        analytical_flux_coeffs = af.moddims(
            analytical_flux_coeffs, d0 = params.N_LGL,
            d1 = params.N_LGL * params.N_Elements * shape_u_n[2],
            d2 = 1, d3 = 1)
        
        analytical_flux_coeffs = af.transpose(analytical_flux_coeffs)

        dl_dxi_coefficients    = af.tile(af.tile(params.dl_dxi_coeffs,
                                                 d0 = params.N_Elements), \
                                         d0 = shape_u_n[2])

        volume_int_coeffs = utils.poly1d_product(dl_dxi_coefficients,
                                                analytical_flux_coeffs)
        
        flux_integral = lagrange.integrate(volume_int_coeffs)
        flux_integral = af.moddims(af.transpose(flux_integral),
                                   d0 = params.N_LGL,
                                   d1 = params.N_Elements,
                                   d2 = shape_u_n[2])

    return flux_integral



def lax_friedrichs_flux(u_n):
    '''
    Calculates the lax-friedrichs_flux :math:`f_i` using.

    :math:`f_i = \\frac{F(u^{i + 1}_0) + F(u^i_{N_{LGL} - 1})}{2} - \\frac
                {\Delta x}{2\Delta t} (u^{i + 1}_0 - u^i_{N_{LGL} - 1})`

    The algorithm used is explained in this `document`_

    .. _document: `https://goo.gl/sNsXXK`


    Parameters
    ----------

    u_n : arrayfire.Array [N_LGL N_Elements M 1]
          Amplitude of the wave at the mapped LGL nodes of each element.
          This code will work for :math:`M` multiple ``u_n``.
    
    Returns
    -------

    boundary_flux : arrayfire.Array [1 N_Elements 1 1]
                    Contains the value of the flux at the boundary elements.
                    Periodic boundary conditions are used.

    '''
    
    u_iplus1_0    = af.shift(u_n[0, :], 0, -1)
    u_i_N_LGL     = u_n[-1, :]
    flux_iplus1_0 = flux_x(u_iplus1_0)
    flux_i_N_LGL  = flux_x(u_i_N_LGL)
    
    boundary_flux = (flux_iplus1_0 + flux_i_N_LGL) / 2 \
                        - params.c_lax * (u_iplus1_0 - u_i_N_LGL) / 2
    
    
    return boundary_flux 



def analytical_u_LGL(t_n):
    '''

    Calculates the analytical u at the LGL points.

    Parameters
    ----------

    t_n : int
          The timestep at which the analytical u is to be calculated.

    Returns
    -------

    u_t_n : arrayfire.Array [N_LGL N_Elements 1 1]
            The value of u at the mapped LGL points in each element.

    '''

    time  = t_n * params.delta_t 
    
    #u_t_n = af.sin(2 * np.pi * (params.element_LGL - params.c * time))
    
    # Multiple waves test
    E_00 = 1.
    E_01 = 1.

    B_00 = 0.2
    B_01 = 0.5

    E_z_t_n = E_00 * af.sin(2 * np.pi * (params.element_LGL - params.c * time))\
            + E_01 * af.cos(2 * np.pi * (params.element_LGL - params.c * time))

    B_y_t_n = B_00 * af.sin(2 * np.pi * (params.element_LGL - params.c * time))\
            + B_01 * af.cos(2 * np.pi * (params.element_LGL - params.c * time))

    u_t_n = B_y_t_n
    
    return u_t_n



def surface_term(u_n):
    '''

    Calculates the surface term,
    :math:`L_p(1) f_i - L_p(-1) f_{i - 1}`
    using the lax_friedrichs_flux function and lagrange_basis_value
    from params module.
    
    Parameters
    ----------
    u_n : arrayfire.Array [N_LGL N_Elements 1 1]
          Amplitude of the wave at the mapped LGL nodes of each element.
          
    Returns
    -------
    surface_term : arrayfire.Array [N_LGL N_Elements 1 1]
                   The surface term represented in the form of an array,
                   :math:`L_p (1) f_i - L_p (-1) f_{i - 1}`, where p varies
                   from zero to :math:`N_{LGL}` and i from zero to
                   :math:`N_{Elements}`. p varies along the rows and i along
                   columns.
    
    **See:** `PDF`_ describing the algorithm to obtain the surface term.
    
    .. _PDF: https://goo.gl/Nhhgzx

    '''

    L_p_minus1   = params.lagrange_basis_value[:, 0]
    L_p_1        = params.lagrange_basis_value[:, -1]
    f_i          = lax_friedrichs_flux(u_n)
    f_iminus1    = af.shift(f_i, 0, 1)
    surface_term = af.blas.matmul(L_p_1, f_i) - af.blas.matmul(L_p_minus1,\
                                                                    f_iminus1)
    
    return surface_term



def surface_term_multiple_u(u_n):
    '''

    Calculates the surface term,
    :math:`L_p(1) f_i - L_p(-1) f_{i - 1}`
    using the lax_friedrichs_flux function and lagrange_basis_value
    from params module.
    
    Parameters
    ----------
    u_n : arrayfire.Array [N_LGL N_Elements M 1]
          Amplitude of the wave at the mapped LGL nodes of each element.
          This code will work for multiple :math:`M` ``u_n``
          
    Returns
    -------
    surface_term : arrayfire.Array [N_LGL N_Elements M 1]
                   The surface term represented in the form of an array,
                   :math:`L_p (1) f_i - L_p (-1) f_{i - 1}`, where p varies
                   from zero to :math:`N_{LGL}` and i from zero to
                   :math:`N_{Elements}`. p varies along the rows and i along
                   columns.
    
    **See:** `PDF`_ describing the algorithm to obtain the surface term.
    
    .. _PDF: https://goo.gl/Nhhgzx

    '''

    shape_u_n = utils.shape(u_n)

    L_p_minus1   = af.tile(params.lagrange_basis_value[:, 0],
                        d0 = 1, d1 = 1, d2 = shape_u_n[2])
    L_p_1        = af.tile(params.lagrange_basis_value[:, -1],
                        d0 = 1, d1 = 1, d2 = shape_u_n[2])
    f_i          = lax_friedrichs_flux(u_n)
    f_iminus1    = af.shift(f_i, 0, 1)

    surface_term = utils.matmul_3D(L_p_1, f_i) \
                 - utils.matmul_3D(L_p_minus1, f_iminus1)
    
    return surface_term



def b_vector(u_n):
    '''

    Calculates the b vector for N_Elements number of elements.
    
    Parameters
    ----------

    u_n : arrayfire.Array [N_LGL N_Elements 1 1]
          Amplitude of the wave at the mapped LGL nodes of each element.

    Returns
    -------

    b_vector_array : arrayfire.Array [N_LGL N_Elements 1 1]
                     Contains the b vector(of shape [N_LGL 1 1 1])
                     for each element.

    **See:** `Report`_ for the b-vector can be found here

    .. _Report: https://goo.gl/sNsXXK

    '''
    volume_integral = volume_integral_flux_multiple_u_n(u_n)
    Surface_term    = surface_term_multiple_u(u_n)    
    b_vector_array  = (volume_integral - Surface_term)
    
    return b_vector_array




def RK4_timestepping(A_inverse, u, delta_t):
    '''

    Implementing the Runge-Kutta (RK4) method to evolve the wave.

    Parameters
    ----------
    A_inverse : arrayfire.Array[N_LGL N_LGL 1 1]
                The inverse of the A matrix which was calculated
                using A_matrix() function.

    u         : arrayfire.Array[N_LGL N_Elements 1 1]
                u at the mapped LGL points

    delta_t   : float64
                The time-step by which u is to be evolved.

    Returns
    -------
    delta_u : arrayfire.Array [N_LGL N_Elements 1 1]
              The change in u at the mapped LGL points.

    '''

    k1 = af.matmul(A_inverse, b_vector(u))
    k2 = af.matmul(A_inverse, b_vector(u + k1 * delta_t / 2))
    k3 = af.matmul(A_inverse, b_vector(u + k2 * delta_t / 2))
    k4 = af.matmul(A_inverse, b_vector(u + k3 * delta_t))

    delta_u = delta_t * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return delta_u



def RK4_timestepping_multiple_u(A_inverse, u, delta_t):
    '''

    Implementing the Runge-Kutta (RK4) method to evolve the wave.

    Parameters
    ----------
    A_inverse : arrayfire.Array[N_LGL N_LGL 1 1]
                The inverse of the A matrix which was calculated
                using A_matrix() function.

    u         : arrayfire.Array[N_LGL N_Elements M 1]
                u at the mapped LGL points

    delta_t   : float64
                The time-step by which u is to be evolved.

    Returns
    -------
    delta_u : arrayfire.Array [N_LGL N_Elements 1 1]
              The change in u at the mapped LGL points.
    '''

    k1 = utils.matmul_3D(A_inverse, b_vector(u))
    k2 = utils.matmul_3D(A_inverse, b_vector(u + k1 * delta_t / 2))
    k3 = utils.matmul_3D(A_inverse, b_vector(u + k2 * delta_t / 2))
    k4 = utils.matmul_3D(A_inverse, b_vector(u + k3 * delta_t))

    delta_u = delta_t * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return delta_u


def RK6_timestepping(A_inverse, u, delta_t):
    '''

    Implementing the Runge-Kutta (RK4) method to evolve the wave.

    Parameters
    ----------
    A_inverse : arrayfire.Array[N_LGL N_LGL 1 1]
                The inverse of the A matrix which was calculated
                using A_matrix() function.

    u         : arrayfire.Array[N_LGL N_Elements 1 1]
                u at the mapped LGL points

    delta_t   : float64
                The time-step by which u is to be evolved.

    Returns
    -------
    delta_u : arrayfire.Array [N_LGL N_Elements 1 1]
              The change in u at the mapped LGL points.

    '''

    k1 = af.matmul(A_inverse, b_vector(u                      ))
    k2 = af.matmul(A_inverse, b_vector(u + 0.25 * k1 * delta_t))
    k3 = af.matmul(A_inverse, b_vector(u + (3 / 32)\
                                         * (k1 \
                                         + 3 * k2)\
                                         * delta_t,             ))
    k4 = af.matmul(A_inverse, b_vector(u + (12 / 2197)\
                                         * (161 * k1\
                                         - 600 * k2 \
                                         + 608 * k3)\
                                         * delta_t,             ))
    k5 = af.matmul(A_inverse, b_vector(u + (1 / 4104)\
                                         * (8341 * k1\
                                         - 32832 * k2\
                                         + 29440 * k3\
                                         - 845 * k4)\
                                         * delta_t,             ))
    k6 = af.matmul(A_inverse, b_vector(u + (-(8/27) * k1\
                                         + 2 * k2\
                                         - (3544 / 2565) * k3\
                                         + (1859 / 4104) * k4\
                                         - (11 / 40) * k5)\
                                         * delta_t,             ))

    delta_u = delta_t * 1 / 5 * (   (16 / 27)       * k1\
                                  + (6656 / 2565)   * k3\
                                  + (28561 / 11286) * k4\
                                  - (9 / 10)        * k5\
                                  + (2 / 11)        * k6\
                                )

    return delta_u



def time_evolution(u = None):
    '''

    Solves the wave equation
    :math: `u^{t_n + 1} = b(t_n) \\times A`
    iterated over time.shape[0] time steps t_n 

    Second order time stepping is used.
    It increases the accuracy of the wave evolution.

    The second order time-stepping would be
    `U^{n + 1/2} = U^n + dt / 2 (A^{-1} B(U^n))`
    `U^{n + 1}   = U^n + dt     (A^{-1} B(U^{n + 1/2}))`
    
    Returns
    -------

    u_diff : arrayfire.Array [N_LGL N_Elements 1 1]
             The absolute of the difference between the numerical
             and analytical value of u at the LGL points.

    '''

    # Creating a folder to store hdf5 files. If it doesn't exist.
    results_directory = 'results/hdf5_%02d' %(int(params.N_LGL))
            
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)


    element_LGL = params.element_LGL
    delta_t     = params.delta_t
    #u           = params.u_init
    shape_u_n = utils.shape(u)
    time        = params.time
    A_inverse = af.tile(af.inverse(A_matrix()),
                        d0 = 1, d1 = 1,
                        d2 = shape_u_n[2])

    element_boundaries = af.np_to_af_array(params.np_element_array)
    
    for t_n in trange(0, time.shape[0]):

        # Storing u at timesteps which are multiples of 100.
        if (t_n % 5) == 0:
            h5file = h5py.File('results/hdf5_%02d/dump_timestep_%06d' %(int(params.N_LGL), int(t_n)) + '.hdf5', 'w')
            Ez_dset   = h5file.create_dataset('E_z', data = u[:, :, 0],
                                              dtype = 'd')
            By_dset   = h5file.create_dataset('B_y', data = u[:, :, 1],
                                              dtype = 'd')

            Ez_dset[:, :] = u[:, :, 0]
            By_dset[:, :] = u[:, :, 1]

       # # Implementing second order time-stepping.
       # u_n_plus_half =  u + af.matmul(A_inverse, b_vector(u))\
       #                      * delta_t / 2

       # u            +=  af.matmul(A_inverse, b_vector(u_n_plus_half))\
       #                  * delta_t

        # Implementing RK 4 scheme
        u += RK4_timestepping_multiple_u(A_inverse, u, delta_t)

        # Implementing RK 6 scheme
        #u += RK6_timestepping(A_inverse, u, delta_t)

    u_analytical = analytical_u_LGL(t_n + 1)

    #u_diff = af.abs(u[:, :, 1] - u_analytical)


    #return u_diff

