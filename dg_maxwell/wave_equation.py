#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os

import arrayfire as af
af.set_backend('cpu')
import numpy as np
from matplotlib import pyplot as plt
from tqdm import trange

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


def dx_dxi_numerical(x_nodes, xi):
    '''
    Differential :math:`\\frac{dx}{d \\xi}` calculated by central
    differential method about xi using the isoparam.isoparam_1D function.

    Parameters
    ----------
    
    x_nodes : arrayfire.Array [N_Elements 1 1 1]
              Contains the nodes of elements.
    
    xi      : arrayfire.Array [N_LGL 1 1 1]
              Values of :math:`\\xi`
    
    Returns
    -------
    dx_dxi : arrayfire.Array [N_Elements 1 1 1]
             :math:`\\frac{dx}{d \\xi}`.
    '''
    dxi = 1e-7
    x2  = isoparam.isoparam_1D(x_nodes, xi + dxi)
    x1  = isoparam.isoparam_1D(x_nodes, xi - dxi)
    
    dx_dxi = (x2 - x1) / (2 * dxi)
    
    return dx_dxi


def dx_dxi_analytical(x_nodes, xi):
    '''
    The analytical result for :math:`\\frac{dx}{d \\xi}` for a 1D element is
    :math:`\\frac{x_1 - x_0}{2}`

    Parameters
    ----------
    x_nodes : arrayfire.Array [N_Elements 1 1 1]
              Contains the nodes of elements.
 
    xi      : arrayfire.Array [N_LGL 1 1 1]
              Values of :math:`\\xi`.

    Returns
    -------
    analytical_dx_dxi : arrayfire.Array [N_Elements 1 1 1]
                        The analytical solution of :math:`\\frac{dx}{d\\xi}`
                        for an element.
    
    '''
    analytical_dx_dxi = (x_nodes[1] - x_nodes[0]) / 2
    
    return analytical_dx_dxi


def A_matrix():
    '''
    Calculates A matrix whose elements :math:`A_{p i}` are given by
    :math:`A_{p i} = \\int^1_{-1} L_p(\\xi)L_i(\\xi) \\frac{dx}{d\\xi}`

    The integrals are computed using the Integrate() function.

    Since elements are taken to be of equal size, :math:`\\frac{dx}{d\\xi}`
    is same everywhere
    

    Returns
    -------
    A_matrix : arrayfire.Array [N_LGL N_LGL 1 1]
               The value of integral of product of lagrange basis functions
               obtained by LGL points, using the Integrate() function
    '''
    int_Li_Lp = lagrange.Integrate(params.lagrange_product)
    dx_dxi    = af.mean(dx_dxi_numerical((params.element_mesh_nodes[0 : 2]),\
                                                        params.xi_LGL))

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


def volume_integral_flux(u_n):
    '''
    Calculates the volume integral of flux in the wave equation.
    :math:`\\int_{-1}^1 f(u) \\frac{d L_p}{d\\xi} d\\xi`
    This will give N values of flux integral as p varies from 0 to N - 1.
    
    This integral is carried out using the analytical form of the integrand
    This integrand is the used in the Integrate() function.

    Calculation of volume integral flux using 8 Lobatto quadrature points
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
    if(params.volume_integral_scheme == 'lobatto_quadrature'\
        and params.N_quad == params.N_LGL):

        integrand       = params.volume_integrand_8_LGL
        lobatto_nodes   = params.lobatto_quadrature_nodes
        Lobatto_weights = params.lobatto_weights_quadrature

        nodes_tile   = af.transpose(af.tile(lobatto_nodes, 1, integrand.shape[1]))
        power        = af.flip(af.range(integrand.shape[1]))
        power_tile   = af.tile(power, 1, params.N_quad)
        nodes_power  = nodes_tile ** power_tile
        weights_tile = af.transpose(af.tile(Lobatto_weights, 1, integrand.shape[1]))
        nodes_weight = nodes_power * weights_tile

        value_at_lobatto_nodes = af.matmul(integrand, nodes_weight)
        F_u_n                  = af.reorder(u_n, 2, 0, 1)
        integral_expansion     = af.broadcast(utils.multiply, value_at_lobatto_nodes, F_u_n)
        flux_integral          = af.sum(integral_expansion, 1)
        flux_integral          = af.reorder(flux_integral, 0, 2, 1)

    else:
        analytical_form_flux       = flux_x(lagrange.wave_equation_lagrange(u_n))
        differential_lagrange_poly = params.differential_lagrange_polynomial

        integrand = np.zeros(([params.N_LGL * params.N_Elements, 2 * params.N_LGL - 2]))

        for i in range(params.N_LGL):
            for j in range(params.N_Elements):
                integrand[i + params.N_LGL * j] = (analytical_form_flux[j] *\
                                                  differential_lagrange_poly[i]).c

        integrand     = af.np_to_af_array(integrand)
        flux_integral = lagrange.Integrate(integrand)
        flux_integral = af.moddims(flux_integral, params.N_LGL, params.N_Elements)

    return flux_integral


def lax_friedrichs_flux(u_n):
    '''
    Calculates the lax-friedrichs_flux :math:`f_i` using.
    
    .. math:: f_i = \\frac{F(u^{i + 1}_0) + F(u^i_{N_{LGL} - 1})}{2} \\\\
        - \\frac{\Delta x}{2\Delta t} (u^{i + 1}_0 - u^i_{N_{LGL} - 1})

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
    volume_integral = volume_integral_flux(u_n)
    Surface_term    = surface_term(u_n)
    b_vector_array  = params.delta_t * (volume_integral - Surface_term)
    
    return b_vector_array


def time_evolution():
    '''
    Solves the wave equation
    :math:`u^{t_n + 1} = b(t_n) \\times A`
    iterated over time steps t_n and then plots :math:`x` against the amplitude
    of the wave. The images are then stored in Wave folder.
    '''
    A_inverse   = af.inverse(A_matrix())
    delta_t     = params.delta_t
    amplitude   = params.u
    time        = params.time
    
    for t_n in trange(0, time.shape[0] - 1):
        
        amplitude[:, :, t_n + 1] =   amplitude[:, :, t_n]\
                                   + af.blas.matmul(A_inverse,\
                                     b_vector(amplitude[:, :, t_n]))
    
    print('u calculated!')
    
    results_directory = 'results/1D_Wave_images'
    
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)

    for t_n in trange(0, time.shape[0] - 1):
        
        if t_n % 100 == 0:
            
            fig = plt.figure()
            x   = params.element_LGL
            y   = amplitude[:, :, t_n]
            
            plt.plot(x, y)
            plt.xlabel('x')
            plt.ylabel('Amplitude')
            plt.title('Time = %f' % (t_n * delta_t))
            fig.savefig('results/1D_Wave_images/%04d' %(t_n / 100) + '.png')
            plt.close('all')
                
    return
