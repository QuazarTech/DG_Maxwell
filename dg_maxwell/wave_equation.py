#! /usr/bin/env python3
# -*- coding: utf-8 -*-
    
import os

import os

import arrayfire as af
af.set_backend('cpu')
import numpy as np
from matplotlib import pyplot as plt
from tqdm import trange
import h5py

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

    x_nodes : arrayfire.Array [N_Elements 1 1 1]
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

    The integrals are computed using the Integrate() function.
    Since elements are taken to be of equal size, :math: `\\frac {dx}{dxi}
    is same everywhere
    
    Returns
    -------

    A_matrix : arrayfire.Array [N_LGL N_LGL 1 1]
               The value of integral of product of lagrange basis functions
               obtained by LGL points, using the Integrate() function

    '''
    int_Li_Lp = lagrange.Integrate(params.lagrange_product)
    dx_dxi    = params.dx_dxi 

    A_matrix_flat = dx_dxi * int_Li_Lp
    A_matrix      = af.moddims(A_matrix_flat, params.N_LGL, params.N_LGL)
    
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
    flux = params.c * u

    return flux


def volume_integral_flux(u_n, t_n):
    '''

    Calculates the volume integral of flux in the wave equation.

    :math:`\\int_{-1}^1 f(u) \\frac{d L_p}{d\\xi} d\\xi`

    This will give N values of flux integral as p varies from 0 to N - 1.
    
    This integral is carried out using the analytical form of the integrand
    obtained as a linear combination of Lagrange basis polynomials.

    This integrand is the used in the Integrate() function.

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
    diff_lag_coeff  = params.volume_integrand_N_LGL

    lobatto_nodes   = params.lobatto_quadrature_nodes
    Lobatto_weights = params.lobatto_weights_quadrature

    nodes_tile   = af.transpose(af.tile(lobatto_nodes, 1, diff_lag_coeff.shape[1]))
    power        = af.flip(af.range(diff_lag_coeff.shape[1]))
    power_tile   = af.tile(power, 1, params.N_quad)
    nodes_power  = nodes_tile ** power_tile
    weights_tile = af.transpose(af.tile(Lobatto_weights, 1, diff_lag_coeff.shape[1]))
    nodes_weight = nodes_power * weights_tile

    dLp_dxi      = af.matmul(diff_lag_coeff, nodes_weight)


    
    if(params.volume_integral_scheme == 'lobatto_quadrature'\
        and params.N_quad == params.N_LGL):

        # Flux using u_n, reordered to 1 X N_LGL X N_Elements array.
        F_u_n                  = af.reorder(flux_x(u_n), 2, 0, 1)

        # Multiplying with dLp / d\xi
        integral_expansion     = af.broadcast(utils.multiply,\
                                 dLp_dxi, F_u_n)

        # Using the quadrature rule.
        flux_integral          = af.sum(integral_expansion, 1)
        flux_integral          = af.reorder(flux_integral, 0, 2, 1)

    elif(params.volume_integral_scheme == 'analytical'):
        print('option2')

        # u_n calculated using af.sin
        analytical_u_n         = analytical_u_LGL(t_n)
        F_u_n                  = af.reorder(analytical_u_n, 2, 0, 1)


        # Multiplying with dLp / d\xi
        integral_expansion     = af.broadcast(utils.multiply, dLp_dxi, F_u_n)

        # Using the quadrature rule.
        flux_integral          = af.sum(integral_expansion, 1)
        flux_integral = af.reorder(flux_integral, 0, 2, 1)


    else:
        # Obtaining the u_n in polynomial form using the value at LGL points.
        analytical_form_flux       = flux_x(lagrange.\
                                            wave_equation_lagrange(u_n))

        # dLp/ d\xi in poly1d form
        differential_lagrange_poly = params.differential_lagrange_polynomial


        # The coefficients of the integrand dLp / dxi * F(u)
        # arranged in a (N_LGL * N_Elements) X (N_LGL - 2) array
        integrand_coeff = np.zeros(([params.N_LGL * params.N_Elements,\
                                          2 * params.N_LGL - 2]))

        for i in range(params.N_LGL):
            for j in range(params.N_Elements):
                # Coefficients of the integrand.
                integrand_coeff[i + params.N_LGL * j] = (analytical_form_flux[j] *\
                                                  differential_lagrange_poly[i]).c

        integrand_coeff = af.np_to_af_array(integrand_coeff)
        flux_integral   = lagrange.Integrate(integrand_coeff)
        flux_integral   = af.moddims(flux_integral, params.N_LGL,\
                                             params.N_Elements) * params.c




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

    u_n : arrayfire.Array [N_LGL N_Elements 1 1]
          Amplitude of the wave at the mapped LGL nodes of each element.
    
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
    u_t_n = af.sin(2 * np.pi * (params.element_LGL - params.c * time)) 

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


def b_vector(u_n, t_n):
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
    volume_integral = volume_integral_flux(u_n, t_n)
    Surface_term    = surface_term(u_n)
    b_vector_array  = (volume_integral - Surface_term)
    
    return b_vector_array

def RK4_timestepping(A_inverse, u, t_n, delta_t):
    '''
    '''

    k1 = af.matmul(A_inverse, b_vector(u                   , t_n))
    k2 = af.matmul(A_inverse, b_vector(u + k1 * delta_t / 2, t_n))
    k3 = af.matmul(A_inverse, b_vector(u + k2 * delta_t / 2, t_n))
    k4 = af.matmul(A_inverse, b_vector(u + k3 * delta_t    , t_n))

    delta_u = delta_t * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return delta_u

def RK6_timestepping(A_inverse, u, t_n, delta_t):
    '''
    '''

    k1 = af.matmul(A_inverse, b_vector(u, t_n))
    k2 = af.matmul(A_inverse, b_vector(u + 0.25 * k1 * delta_t, t_n))
    k3 = af.matmul(A_inverse, b_vector(u + (3 / 32) * (k1 + 3 * k2) * delta_t, t_n))
    k4 = af.matmul(A_inverse, b_vector(u + (12 / 2197) * (161 * k1 - 600 * k2 + 608 * k3) * delta_t, t_n))
    k5 = af.matmul(A_inverse, b_vector(u + (1 / 4104) * (8341 * k1 - 32832 * k2 + 29440 * k3 - 845 * k4) * delta_t, t_n))
    k6 = af.matmul(A_inverse, b_vector(u + (-(8/27) * k1 + 2 * k2- (3544 / 2565) *\
                                       k3 + (1859 / 4104) * k4 - (11 / 40) * k5) * delta_t, t_n))

    delta_u = delta_t * 1 / 5 * ((16 / 27) * k1 + (6656 / 2565) * k3 + (28561 / 11286) * k4 - (9 / 10) * k5 + (2 / 11) * k6)

    return delta_u


def time_evolution():
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

    L1_norm : float64 
              The L1 norm of error after one full advection.

    '''
    # Creating a folder to store hdf5 files. If it doesn't exist.
    results_directory = 'results/hdf5_%02d' %int(params.N_LGL)
            
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)

    A_inverse   = af.inverse(A_matrix())
    element_LGL = params.element_LGL
    delta_t     = params.delta_t
    u           = params.u_init
    time        = params.time

    element_boundaries = af.np_to_af_array(params.np_element_array)

    for t_n in trange(0, time.shape[0]):

        # Storing u at timesteps which are multiples of 20.
        if (t_n % 50) == 0:
            h5file = h5py.File('results/hdf5_%02d/dump_timestep_%06d' %(int(params.N_LGL), int(t_n)) + '.hdf5', 'w')
            dset   = h5file.create_dataset('u_i', data = u, dtype = 'd')

            dset[:, :] = u[:, :]

       # # Implementing second order time-stepping.
       # u_n_plus_half =  u + af.matmul(A_inverse, b_vector(u, t_n))\
       #                      * delta_t / 2

       # u            +=  af.matmul(A_inverse, b_vector(u_n_plus_half, t_n + 0.5))\
       #                  * delta_t

        # Implementing RK 4 scheme
        #u += RK4_timestepping(A_inverse, u, t_n, delta_t)

        # Implementing RK 6 scheme
        u += RK6_timestepping(A_inverse, u, t_n, delta_t)

    u_analytical = analytical_u_LGL(t_n + 1)



    return (af.abs(u - u_analytical))


def change_parameters(LGL, Elements, wave='sin'):
    '''

    Changes the parameters of the simulation. Used only for convergence tests.

    Parameters
    ----------
    LGL      : int
               The new N_LGL.

    Elements : int
               The new N_Elements.

    '''
    # The domain of the function.
    params.x_nodes    = af.np_to_af_array(np.array([-1., 1.]))

    # The number of LGL points into which an element is split.
    params.N_LGL      = LGL

    # Number of elements the domain is to be divided into.
    params.N_Elements = Elements

    # The number quadrature points to be used for integration.
    params.N_quad     = LGL + 1

    # Array containing the LGL points in xi space.
    params.xi_LGL     = lagrange.LGL_points(params.N_LGL)

    # N_Gauss number of Gauss nodes.
    params.gauss_points  = af.np_to_af_array(lagrange.gauss_nodes\
                                                    (params.N_quad))

    # The Gaussian weights.
    params.gauss_weights = lagrange.gaussian_weights(params.N_quad)

    # The lobatto nodes to be used for integration.
    params.lobatto_quadrature_nodes   = lagrange.LGL_points(params.N_quad)

    # The lobatto weights to be used for integration.
    params.lobatto_weights_quadrature = lagrange.lobatto_weights\
                                        (params.N_quad)

    # A list of the Lagrange polynomials in poly1d form.
    params.lagrange_product = lagrange.product_lagrange_poly(params.xi_LGL)

    # An array containing the coefficients of the lagrange basis polynomials.
    params.lagrange_coeffs  = af.np_to_af_array(\
                              lagrange.lagrange_polynomials(params.xi_LGL)[1])

    # Refer corresponding functions.
    params.lagrange_basis_value = lagrange.lagrange_function_value\
                                           (params.lagrange_coeffs)

    # A list of the Lagrange polynomials in poly1d form.
    params.lagrange_poly1d_list = lagrange.lagrange_polynomials(params.xi_LGL)[0]


    # list containing the poly1d forms of the differential of Lagrange
    # basis polynomials.
    params.differential_lagrange_polynomial = lagrange.differential_lagrange_poly1d()


    # While evaluating the volume integral using N_LGL
    # lobatto quadrature points, The integration can be vectorized
    # and in this case the coefficients of the differential of the
    # Lagrange polynomials is required
    params.volume_integrand_N_LGL = np.zeros(([params.N_LGL, params.N_LGL - 1]))

    for i in range(params.N_LGL):
        params.volume_integrand_N_LGL[i] = (params.differential_lagrange_polynomial[i]).c

    params.volume_integrand_N_LGL= af.np_to_af_array(params.volume_integrand_N_LGL)

    # Obtaining an array consisting of the LGL points mapped onto the elements.

    params.element_size    = af.sum((params.x_nodes[1] - params.x_nodes[0])\
                                                        / params.N_Elements)
    params.elements_xi_LGL = af.constant(0, params.N_Elements, params.N_LGL)
    params.elements        = utils.linspace(af.sum(params.x_nodes[0]),
                             af.sum(params.x_nodes[1] - params.element_size),\
                                                            params.N_Elements)

    params.np_element_array   = np.concatenate((af.transpose(params.elements),
                                   af.transpose(params.elements +\
                                                       params.element_size)))

    params.element_mesh_nodes = utils.linspace(af.sum(params.x_nodes[0]),
                                        af.sum(params.x_nodes[1]),\
                                               params.N_Elements + 1)

    params.element_array = af.transpose(af.np_to_af_array\
                                       (params.np_element_array))
    params.element_LGL   = mapping_xi_to_x(af.transpose\
                                          (params.element_array), params.xi_LGL)

    # The minimum distance between 2 mapped LGL points.
    params.delta_x = af.min((params.element_LGL - af.shift(params.element_LGL, 1, 0))[1:, :])

    # dx_dxi for elements of equal size.
    params. dx_dxi = af.mean(dx_dxi_numerical((params.element_mesh_nodes[0 : 2]),\
                                   params.xi_LGL))


    # The value of time-step.
    params.delta_t = params.delta_x / (4 * params.c)

    # Array of timesteps seperated by delta_t.
    params.time    = utils.linspace(0, int(params.total_time / params.delta_t) * params.delta_t,
                                                        int(params.total_time / params.delta_t))

    # Initializing the amplitudes. Change u_init to required initial conditions.
    if (wave=='sin'):
        params.u_init     = af.sin(2 * np.pi * params.element_LGL)
        
    if (wave=='gaussian'):
        params.u_init = np.e ** (-(params.element_LGL) ** 2 / 0.4 ** 2)
    
    params.u          = af.constant(0, params.N_LGL, params.N_Elements, params.time.shape[0],\
                                     dtype = af.Dtype.f64)
    params.u[:, :, 0] = params.u_init
                                                     
    return

def L1_norm_error(mod_diff_u):
    '''
    '''
    wave_equation_coeffs = np.zeros([params.N_Elements, params.N_LGL])

    for i in range(0, params.N_Elements):
        wave_equation_coeffs[i, :] = lagrange.wave_equation_lagrange(mod_diff_u)[i].c

    L1_norm = af.sum(lagrange.Integrate(af.np_to_af_array\
                                       (wave_equation_coeffs))) * params.dx_dxi

    return L1_norm


def convergence_test():
    '''

    Used to obtain plots of L1 norm versus parameters (Number of elements
    or N_LGL).
    
    '''
    L1_norm_option_1 = np.zeros([10])
    N_lgl            = (np.arange(10) + 3).astype(float)

    for i in range(0, 10):
        change_parameters(i + 3, 64)
        u_diff = time_evolution()
        L1_norm_option_1[i] = L1_norm_error(u_diff)

#    L1_norm_option_1 = np.array([7.86867473e-03, 1.62010118e-04, 6.09539662e-06, 3.33291808e-07,\
#                                 1.60409746e-07, 6.26348505e-07, 3.40978741e-07, 3.06485997e-07,\
#                                 1.97638417e-07, 2.09291363e-08])


    print(L1_norm_option_1)
    normalization = 0.00786 / (3 ** (-3))
    plt.loglog(N_lgl, L1_norm_option_1, marker='o', label='L1 norm')
    plt.xlabel('No. of LGL points')
    plt.ylabel('L1 norm of error')
    plt.title('L1 norm after 10 full advections')
    plt.loglog(N_lgl, normalization * N_lgl **(-N_lgl), color='black', linestyle='--', label='$N_{LGL}^{-N_{LGL}}$')
    plt.legend(loc='best')

    plt.show()

    return
