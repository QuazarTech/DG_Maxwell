#! /usr/bin/env python3

import arrayfire as af
af.set_backend('opencl')
import numpy as np
from app import global_variables as gvar
from app import lagrange
from utils import utils
from matplotlib import pyplot as plt
import pylab as pl
from tqdm import trange

plt.rcParams['figure.figsize'] = 9.6, 6.
plt.rcParams['figure.dpi'] = 100
plt.rcParams['image.cmap'] = 'jet'
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 20
plt.rcParams['font.sans-serif'] = 'serif'
plt.rcParams['text.usetex'] = True
plt.rcParams['axes.linewidth'] = 1.5
plt.rcParams['axes.titlesize'] = 'medium'
plt.rcParams['axes.labelsize'] = 'medium'
plt.rcParams['xtick.major.size'] = 8
plt.rcParams['xtick.minor.size'] = 4
plt.rcParams['xtick.major.pad'] = 8
plt.rcParams['xtick.minor.pad'] = 8
plt.rcParams['xtick.color'] = 'k'
plt.rcParams['xtick.labelsize'] = 'medium'
plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.major.size'] = 8
plt.rcParams['ytick.minor.size'] = 4
plt.rcParams['ytick.major.pad'] = 8
plt.rcParams['ytick.minor.pad'] = 8
plt.rcParams['ytick.color'] = 'k'
plt.rcParams['ytick.labelsize'] = 'medium'
plt.rcParams['ytick.direction'] = 'in'



def mapping_xi_to_x(x_nodes, xi):
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
    
    N0_x0 = af.broadcast(utils.multiply, N_0, x_nodes[0])
    N1_x1 = af.broadcast(utils.multiply, N_1, x_nodes[1])
    
    x = N0_x0 + N1_x1
    
    return x



def dx_dxi_numerical(x_nodes, xi):
    '''
    Differential :math: `\\frac{dx}{d \\xi}` calculated by central differential
    method about xi using the mapping_xi_to_x function.
    
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
    x2  = mapping_xi_to_x(x_nodes, xi + dxi)
    x1  = mapping_xi_to_x(x_nodes, xi - dxi)
    
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
    dx_dxi          = dx_dxi_numerical((gvar.element_mesh_nodes[0 : 2]),
                                                                   gvar.xi_LGL)
    dx_dxi_tile     = af.tile(dx_dxi, 1, gvar.N_LGL)
    identity_matrix = af.identity(gvar.N_LGL, gvar.N_LGL, dtype = af.Dtype.f64)
    A_matrix        = af.broadcast(utils.multiply, gvar.lobatto_weights,
                                                 identity_matrix) * dx_dxi_tile

    return A_matrix


def flux_x(u):
    '''
    A function which returns the value of flux for a given wave function u.
    :math:`f(u) = c u^k`
    
    Parameters
    ----------
    u    : arrayfire.Array
           A 1-D array which contains the value of wave function.
    
    Returns
    -------
    flux : arrayfire.Array
           The value of the flux for given u.
    '''
    flux = gvar.c * u

    return flux


def volume_integral_flux(u):
    '''
    A function to calculate the volume integral of flux in the wave equation.
    :math:`\\int_{-1}^1 f(u) \\frac{d L_p}{d\\xi} d\\xi`
    This will give N values of flux integral as p varies from 0 to N - 1.
    
    This integral is carried out over an element with LGL nodes mapped onto it.
    
    Parameters
    ----------
    u             : arrayfire.Array [N_LGL N_Elements 1 1]
                    A 1-D array containing the value of the wave function at
                    the mapped LGL nodes in the element.
    
    Returns
    -------
    flux_integral : arrayfire.Array [N_LGL N_Elements 1 1]
                    A 1-D array of the value of the flux integral calculated
                    for various lagrange basis functions.
    '''
    
    dLp_xi        = gvar.dLp_xi
    weight_tile   = af.transpose(af.tile(gvar.lobatto_weights, 1, gvar.N_LGL))
    dLp_xi       *= weight_tile
    flux          = flux_x(u)
    weight_flux   = flux
    flux_integral = af.blas.matmul(dLp_xi, weight_flux)
    
    return flux_integral


def lax_friedrichs_flux(u_t_n):
    '''
    A function which calculates the lax-friedrichs_flux :math:`f_i` using.
    :math:`f_i = \\frac{F(u^{i + 1}_0) + F(u^i_{N_{LGL} - 1})}{2} - \frac
                {\Delta x}{2\Delta t} (u^{i + 1}_0 - u^i_{N_{LGL} - 1})`
    
    Parameters
    ----------
    u_t_n : arrayfire.Array [N_LGL N_Elements 1 1]
            A 2D array consisting of the amplitude of the wave at the LGL nodes
            at each element.
    '''
    
    u_iplus1_0    = af.shift(u_t_n[0, :], 0, -1)
    u_i_N_LGL     = u_t_n[-1, :]
    flux_iplus1_0 = flux_x(u_iplus1_0)
    flux_i_N_LGL  = flux_x(u_i_N_LGL)
    
    boundary_flux = (flux_iplus1_0 + flux_i_N_LGL) / 2 \
                        - gvar.c_lax * (u_iplus1_0 - u_i_N_LGL) / 2
    
    
    return boundary_flux 


def surface_term(u_t_n):
    '''
    A function which is used to calculate the surface term,
    :math:`L_p (1) f_i - L_p (-1) f_{i - 1}`
    using the lax_friedrichs_flux function and lagrange_basis_value
    from gvar module.
    
    Parameters
    ----------
    u_t_n : arrayfire.Array [N M 1 1]
          The timestep at which the surface term is to be calculated.
    
    Returns
    -------
    surface_term : arrayfire.Array [N_LGL N_Elements 1 1]
                   The surface term represented in the form of an array,
                   :math:`L_p (1) f_i - L_p (-1) f_{i - 1}`, where p varies from
                   zero to :math:`N_{LGL}` and i from zero to
                   :math:`N_{Elements}`. p varies along the rows and i along
                   columns.
    
    Reference
    ---------
    Link to PDF describing the algorithm to obtain the surface term.
    
    https://cocalc.com/projects/1b7f404c-87ba-40d0-816c-2eba17466aa8/files
    /PM\_2\_5/wave\_equation/documents/surface\_term/surface\_term.pdf
    '''

    L_p_minus1   = gvar.lagrange_basis_value[:, 0]
    L_p_1        = gvar.lagrange_basis_value[:, -1]
    f_i          = lax_friedrichs_flux(u_t_n)
    f_iminus1    = af.shift(f_i, 0, 1)
    surface_term = af.blas.matmul(L_p_1, f_i) - af.blas.matmul(L_p_minus1,\
                                                                    f_iminus1)
    
    return surface_term


def b_vector(u_t_n):
    '''
    A function which returns the b vector for N_Elements number of elements.
    
    Parameters
    ----------
    t_n            : double
    
    Returns
    -------
    b_vector_array : arrayfire.Array
    '''

    volume_integral = volume_integral_flux(u_t_n)
    Surface_term    = surface_term(u_t_n)
    b_vector_array  = gvar.delta_t * (volume_integral - Surface_term)
    
    return b_vector_array


def time_evolution():
    '''
    Function which solves the wave equation
    :math: `u^{t_n + 1} = b(t_n) \\times A`
    iterated over time steps t_n and then plots :math: `x` against the amplitude
    of the wave. The images are then stored in Wave folder.
    '''
    
    A_inverse   = af.inverse(A_matrix())
    element_LGL = gvar.element_LGL
    delta_t     = gvar.delta_t
    amplitude   = gvar.u 
    time        = gvar.time
    
    for t_n in trange(0, time.shape[0] - 1):
        
        amplitude[:, :, t_n + 1] = amplitude[:, :, t_n] + af.blas.matmul(A_inverse,\
                b_vector(amplitude[:, :, t_n]))
    
    print('u calculated!')
    
    for t_n in trange(0, time.shape[0] - 1):
        
        if t_n % 100 == 0:
            
            fig = plt.figure()
            x   = gvar.element_LGL
            y   = amplitude[:, :, t_n]
            
            plt.plot(x, y)
            plt.xlabel('x')
            plt.ylabel('Amplitude')
            plt.title('Time = %f' % (t_n * delta_t))
            fig.savefig('results/1D_Wave_images/%04d' %(t_n / 100) + '.png')
            plt.close('all')
                
    return

