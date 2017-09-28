#! /usr/bin/env python3

import arrayfire as af
af.set_backend('opencl')
import numpy as np
from matplotlib import pyplot as plt
import pylab as pl
from tqdm import trange

from app import params
from app import lagrange
from utils import utils

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


def volume_integral_flux(u_n, t_n):
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

        nodes_tile   = af.transpose(af.tile(lobatto_nodes, 1,\
                                          integrand.shape[1]))
        power        = af.flip(af.range(integrand.shape[1]))
        power_tile   = af.tile(power, 1, params.N_quad)
        nodes_power  = nodes_tile ** power_tile
        weights_tile = af.transpose(af.tile(Lobatto_weights, 1,\
                       integrand.shape[1]))
        nodes_weight = nodes_power * weights_tile

        value_at_lobatto_nodes = af.matmul(integrand, nodes_weight)
        F_u_n                  = af.reorder(u_n, 2, 0, 1)
        integral_expansion     = af.broadcast(utils.multiply,\
                                 value_at_lobatto_nodes, F_u_n)
        flux_integral          = af.sum(integral_expansion, 1)
        flux_integral          = af.reorder(flux_integral, 0, 2, 1)

    return flux_integral


def lax_friedrichs_flux(u_n):
    '''
    Calculates the lax-friedrichs_flux :math:`f_i` using.
    :math:`f_i = \\frac{F(u^{i + 1}_0) + F(u^i_{N_{LGL} - 1})}{2} - \\frac
                {\Delta x}{2\Delta t} (u^{i + 1}_0 - u^i_{N_{LGL} - 1})`

    The algorithm used is explained in the link given below
    `https://goo.gl/sNsXXK`

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

def amplitude_quadrature_points(t_n):
    '''
    [TODO]-for the sake of convenience, N_LGL = N_quad
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
    
    Reference
    ---------
    Link to PDF describing the algorithm to obtain the surface term.
    
    `https://goo.gl/Nhhgzx`
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

    Reference
    ---------
    A report for the b-vector can be found here
    `https://goo.gl/sNsXXK`
    '''
    volume_integral = volume_integral_flux(u_n, t_n)
    Surface_term    = surface_term(u_n)
    b_vector_array  = (volume_integral - Surface_term)
    
    return b_vector_array


def time_evolution():
    '''
    Solves the wave equation
    :math: `u^{t_n + 1} = b(t_n) \\times A`
    iterated over time steps t_n and then plots :math: `x` against the amplitude
    of the wave.
    '''
    A_inverse   = af.inverse(A_matrix())
    element_LGL = params.element_LGL
    delta_t     = params.delta_t
    amplitude   = params.u 
    time        = params.time
    
    element_boundaries = af.np_to_af_array(params.np_element_array)

    for t_n in trange(0, time.shape[0] - 1):

        amplitude_n_plus_half =  amplitude[:, :, t_n]\
                               + af.matmul(A_inverse,\
                                 b_vector(amplitude[:, :, t_n], t_n))\
                               * delta_t / 2


        amplitude[:, :, t_n + 1] =   amplitude[:, :, t_n]\
                                   + af.matmul(A_inverse,\
                                     b_vector(amplitude_n_plus_half, t_n))\
                                   * delta_t


    print('u calculated!')

    time_one_pass = int(2 / delta_t)

  #  for t_n in range(0, time.shape[0] - 1):
  #      if(t_n == time_one_pass):
  #          fig = plt.figure()
  #          x   = params.element_LGL
  #          y   = amplitude[:, :, t_n]
  #          
  #          y_analytical = amplitude_quadrature_points(t_n)

  #          #plt.plot(af.flat(x), af.flat(y), label='advected wave')
  #          #plt.plot(af.flat(x), af.flat(y_analytical),'k--',  label='analytical solution')
  #          plt.semilogy(af.flat(x), af.flat(af.abs(y - y_analytical)), marker='o')
  #          plt.xlabel('x')
  #          plt.ylabel('$\|u_{num}$(x) - $u_{analytical}$(x)$\|$')
  #          plt.ylim(-2, 2)
  #          plt.title('Spatial plot of error after 1 full advection')
  #          fig.savefig('results/10_advection'+'.png')
  #          plt.show()

   # for t_n in trange(0, time.shape[0] - 1):

   #     if t_n % 100 == 0:
   #         fig = plt.figure()
   #         x   = params.element_LGL
   #         y   = amplitude[:, :, t_n]
   #         
   #         mod_diff = af.sum(af.abs(amplitude[:, :, t_n] - amplitude_quadrature_points(t_n)))
   #         plt.plot(x, y)
   #         plt.xlabel('x')
   #         plt.ylabel('Amplitude')
   #         plt.ylim(-2, 2)
   #         plt.title('Time = %f' % (t_n * delta_t), '     ', 'Error = ', mod_diff)
   #         fig.savefig('results/1D_Wave_images/%04d' %(t_n / 100) + '.png')
   #         plt.close('all') 

   # x   = np.array(element_boundaries)

   # ax = plt.subplot(111)

   # fig = plt.figure()
   # x = (af.range(time.shape[0] - 1)) * delta_t
   # y = af.constant(0, time.shape[0] - 1)


    wave_equation_coeffs = np.zeros([params.N_Elements, params.N_LGL])

    calculated_u = params.u[:, :, time_one_pass]
    analytical_u = amplitude_quadrature_points(time_one_pass)

    for i in range(0, params.N_Elements):
        wave_equation_coeffs[i, :] = (lagrange.wave_equation_lagrange(af.abs(calculated_u - analytical_u))[i].c)

    L1_norm = (af.sum(lagrange.Integrate(af.np_to_af_array(wave_equation_coeffs))))


    return L1_norm 


def change_parameters(LGL, Elements):
    '''
    '''
    # The domain of the function.
    params.x_nodes    = af.np_to_af_array(np.array([-1., 1.]))

    # The number of LGL points into which an element is split.
    params.N_LGL      = LGL

    # Number of elements the domain is to be divided into.
    params.N_Elements = Elements


    # The number quadrature points to be used for integration.
    # [TODO]- refer amplitude_quadrature_points before changing.
    params.N_quad = LGL

    # Array containing the LGL points in xi space.
    params.xi_LGL     = lagrange.LGL_points(params.N_LGL)

    # N_Gauss number of Gauss nodes.
    params.gauss_points  = af.np_to_af_array(lagrange.gauss_nodes(params.N_quad))

    # The Gaussian weights.
    params.gauss_weights = lagrange.gaussian_weights(params.N_quad)

    # The lobatto nodes to be used for integration.
    params.lobatto_quadrature_nodes = lagrange.LGL_points(params.N_quad)

    # The lobatto weights to be used for integration.
    params.lobatto_weights_quadrature = lagrange.lobatto_weights\
                                        (params.N_quad)

    # A list of the Lagrange polynomials in poly1d form.
    params.lagrange_product = lagrange.product_lagrange_poly(params.xi_LGL)

    # An array containing the coefficients of the lagrange basis polynomials.
    params.lagrange_coeffs = af.np_to_af_array(lagrange.lagrange_polynomials(params.xi_LGL)[1])

    # Refer corresponding functions.
    params.lagrange_basis_value = lagrange.lagrange_function_value(params.lagrange_coeffs)

    # A list of the Lagrange polynomials in poly1d form.
    params.lagrange_poly1d_list = lagrange.lagrange_polynomials(params.xi_LGL)[0]


    # list containing the poly1d forms of the differential of Lagrange
    # basis polynomials.
    params.differential_lagrange_polynomial = lagrange.differential_lagrange_poly1d()


    # While evaluating the volume integral using N_LGL
    # lobatto quadrature points, The integration can be vectorized
    # and in this case the coefficients of the differential of the
    # Lagrange polynomials is required
    params.volume_integrand_8_LGL = np.zeros(([params.N_LGL, params.N_LGL - 1]))

    for i in range(params.N_LGL):
        params.volume_integrand_8_LGL[i] = (params.differential_lagrange_polynomial[i]).c

    params.volume_integrand_8_LGL= af.np_to_af_array(params.volume_integrand_8_LGL)

    # Obtaining an array consisting of the LGL points mapped onto the elements.

    params.element_size    = af.sum((params.x_nodes[1] - params.x_nodes[0]) / params.N_Elements)
    params.elements_xi_LGL = af.constant(0, params.N_Elements, params.N_LGL)
    params.elements        = utils.linspace(af.sum(params.x_nodes[0]),
                      af.sum(params.x_nodes[1] - params.element_size), params.N_Elements)

    params.np_element_array   = np.concatenate((af.transpose(params.elements),
                                   af.transpose(params.elements + params.element_size)))

    params.element_mesh_nodes = utils.linspace(af.sum(params.x_nodes[0]),
                                        af.sum(params.x_nodes[1]), params.N_Elements + 1)

    params.element_array = af.transpose(af.interop.np_to_af_array(params.np_element_array))
    params.element_LGL   = mapping_xi_to_x(af.transpose(params.element_array),\
                                                                       params.xi_LGL)

    # The minimum distance between 2 mapped LGL points.
    params.delta_x = af.min((params.element_LGL - af.shift(params.element_LGL, 1, 0))[1:, :])

    # The value of time-step.
    params.delta_t = params.delta_x / (10 * params.c)

    # Array of timesteps seperated by delta_t.
    params.time    = utils.linspace(0, int(params.total_time / params.delta_t) * params.delta_t,
                                                        int(params.total_time / params.delta_t))

    # Initializing the amplitudes. Change u_init to required initial conditions.
    params.u_init     = af.sin(2 * np.pi * params.element_LGL)#np.e ** (-(element_LGL) ** 2 / 0.4 ** 2)
    params.u          = af.constant(0, params.N_LGL, params.N_Elements, params.time.shape[0],\
                                     dtype = af.Dtype.f64)
    params.u[:, :, 0] = params.u_init
                                                     
    return

def convergence_test():
    '''
    '''
    fig = plt.figure()

    int_u_calculated = np.zeros([22])
    index            = np.arange(22) + 3

    for i in range(3, 25):
        change_parameters(8, i)
        int_u_calculated[i - 3] = (time_evolution())

    L1_norm = int_u_calculated 
    plt.semilogy(index, L1_norm, marker='o')
    plt.xlabel('Number of Elements')
    plt.ylabel('L1 norm of error')
    plt.title('L1 norm v/s No. of Elements')
    plt.show()
    fig.savefig('results/L1_norm.png')

    return
