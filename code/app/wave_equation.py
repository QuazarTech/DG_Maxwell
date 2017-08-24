#! /usr/bin/env python3

from os import sys
import numpy as np
from matplotlib import pyplot as plt
import subprocess

import arrayfire as af
af.set_backend('cuda')
from tqdm import trange

from app import lagrange
from utils import utils
from app import global_variables as gvar

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


def Li_Lp_xi(L_i_xi, L_p_xi):
    '''
    Parameters
    ----------
    L_i_xi : arrayfire.Array [1 N N 1]
                A 2D array :math:`L_i` obtained at LGL points calculated at the
                LGL nodes :math:`N_LGL`.

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
    N0_x0 + N1_x1 : arrayfire.Array
                    :math: `x` value in the element corresponding to
                    :math:`\\xi`.
    '''
    N_0 = (1 - xi) / 2
    N_1 = (1 + xi) / 2

    N0_x0 = af.bcast.broadcast(utils.multiply, N_0, x_nodes[0])
    N1_x1 = af.bcast.broadcast(utils.multiply, N_1, x_nodes[1])

    return N0_x0 + N1_x1


def dx_dxi_numerical(x_nodes, xi):
    '''
    Differential calculated by central differential method about :math: `\\xi`
    using the mappingXiToX function.

    Parameters
    ----------

    x_nodes : arrayfire.Array
                Contains the nodes of elements

    xi		: float
                Value of :math: `\\xi`

    Returns
    -------
    (x2 - x1) / (2 * dxi) : arrayfire.Array
                            :math:`\\frac{dx}{d \\xi}`. 
    '''
    dxi = 1e-7
    x2 = mappingXiToX(x_nodes, xi + dxi)
    x1 = mappingXiToX(x_nodes, xi - dxi)

    return (x2 - x1) / (2 * dxi)


def dx_dxi_analytical(x_nodes, xi):
    '''

    Parameters
    ----------
    x_nodes : arrayfire.Array
                An array containing the nodes of an element.

    Returns
    -------
    (x_nodes[1] - x_nodes[0]) / 2 : arrayfire.Array
                                    The analytical solution of
                                    \\frac{dx}{d\\xi} for an element.

    '''
    return (x_nodes[1] - x_nodes[0]) / 2


def A_matrix():
    '''
    Calculates the value of lagrange basis functions obtained for :math: `N_LGL`
    points at the LGL nodes.

    Returns
    -------
    A_matrix : arrayfire.Array
                The value of integral of product of lagrange basis functions
                obtained by LGL points, using Gauss-Lobatto quadrature method
                using :math: `N_LGL` points. 

    [NOTE]:

    The A matrix will vary for each element. The one calculatedis for the case
    of 1D elements which are of equal size.
    '''
    
    x_tile          = af.transpose(af.tile(gvar.xi_LGL, 1, gvar.N_LGL))
    power           = af.flip(af.range(gvar.N_LGL))
    power_tile      = af.tile(power, 1, gvar.N_LGL)
    x_pow           = af.arith.pow(x_tile, power_tile)
    lobatto_weights = gvar.lobatto_weights

    lobatto_weights_tile = af.tile(af.reorder(lobatto_weights, 1, 2, 0),\
                                                gvar.N_LGL, gvar.N_LGL)

    index = af.range(gvar.N_LGL)
    L_i   = af.blas.matmul(gvar.lBasisArray[index], x_pow)
    L_p   = af.reorder(L_i, 0, 2, 1)
    L_i   = af.reorder(L_i, 2, 0, 1)

    dx_dxi      = dx_dxi_numerical((gvar.elementMeshNodes[0 : 2]), gvar.xi_LGL)
    dx_dxi_tile = af.tile(dx_dxi, 1, gvar.N_LGL, gvar.N_LGL)
    Li_Lp_array = Li_Lp_xi(L_p, L_i)
    L_element   = (Li_Lp_array * lobatto_weights_tile * dx_dxi_tile)
    A_matrix    = af.sum(L_element, dim = 2)

    return A_matrix


def flux_x(u):
    '''
    A function which returns the value of flux for a given wave function u.
    :math:`f(u) = c u^k`

    Parameters
    ----------
    u         : arrayfire.Array
                            A 1-D array which contains the value of wave function.

    Returns
    -------
    c * u : arrayfire.Array
            The value of the flux for given u.
    '''
    return gvar.c * u


def volumeIntegralFlux(element_LGL, u):
    '''
    A function to calculate the volume integral of flux in the wave equation.
    :math:`\\int_{-1}^1 f(u) \\frac{d L_p}{d\\xi} d\\xi`
    This will give N values of flux integral as p varies from 0 to N - 1.

    This integral is carried out over an element with LGL nodes mapped onto it.

    Parameters
    ----------
    element_LGL   : arrayfire.Array [N_LGL N_Elements 1 1]
                    A 2-D array consisting of the LGL nodes mapped onto the
                    element's domain.

    u             : arrayfire.Array [N_LGL N_Elements 1 1]
                    A 1-D array containing the value of the wave function at the
                    mapped LGL nodes in the element.

    Returns
    -------
    flux_integral : arrayfire.Array [1 N 1 1]
                    A 1-D array of the value of the flux integral calculated
                    for various lagrange basis functions.
    '''

    dLp_xi        = gvar.dLp_xi
    weight_tile   = af.tile(gvar.lobatto_weights, 1, gvar.N_Elements)
    flux          = flux_x(u)
    weight_flux   = weight_tile * flux
    flux_integral = af.blas.matmul(dLp_xi, weight_flux)

    return flux_integral


def laxFriedrichsFlux(t_n):
    '''
    A function which calculates the lax-friedrichs_flux :math:`f_i` using.
    :math:`f_i = \\frac{F(u^{i + 1}_0) + F(u^i_{N_{LGL} - 1})}{2} - \frac
                    {\Delta x}{2\Delta t} (u^{i + 1}_0 - u^i_{N_{LGL} - 1})`

    Parameters
    ----------
    u    : arrayfire.Array [N_LGL N_Elements 1 1]
            A 2D array consisting of the amplitude of the wave at the LGL nodes
            at each element.

    t_n  : float
            The timestep at which the lax-friedrichs-flux is to be calculated.
    '''

    u_iplus1_0    = af.shift(gvar.u[0, :, t_n], 0, -1)
    u_i_N_LGL     = gvar.u[-1, :, t_n]

    flux_iplus1_0 = flux_x(u_iplus1_0)
    flux_i_N_LGL  = flux_x(u_i_N_LGL)

    laxFriedrichsFlux = (flux_iplus1_0 + flux_i_N_LGL) / 2 \
                        - gvar.c_lax * (u_iplus1_0 - u_i_N_LGL) / 2

    return laxFriedrichsFlux


def surface_term(t_n):
    '''
    A function which is used to calculate the surface term,
    :math:`L_p (1) f_i - L_p (-1) f_{i - 1}`
    using the laxFriedrichsFlux function and the dLp_xi_LGL function in gvar
    module.

    Parameters
    ----------
    t_n : double
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

    https://cocalc.com/projects/1b7f404c-87ba-40d0-816c-2eba17466aa8/files\
    /PM\_2\_5/wave\_equation/documents/surface\_term/surface\_term.pdf
    '''
    L_p_minus1   = gvar.lagrange_basis_function()[:, 0]
    L_p_1        = gvar.lagrange_basis_function()[:, -1]

    f_i          = laxFriedrichsFlux(t_n)
    f_iminus1    = af.shift(f_i, 0, 1)

    surface_term = af.blas.matmul(L_p_1, f_i) - af.blas.matmul(L_p_minus1,
                                                                f_iminus1)

    return surface_term


def b_vector(t_n):
    '''
    A function which returns the b vector for N_Elements number of elements.

    Parameters
    ----------
    t_n : double

    Returns
    -------
    b_vector_array : arrayfire.Array
    '''
    volume_integral = volumeIntegralFlux(gvar.element_LGL, gvar.u[:, :, t_n])
    surfaceTerm     = surface_term(t_n)
    b_vector_array  = gvar.delta_t * (volume_integral - surfaceTerm)


    return b_vector_array


def time_evolution():
    '''
    Function which solves the wave equation
    :math: `u^{t_n + 1} = b(t_n) \\times A`
    iterated over time steps t_n and then plots :math: `x` against the amplitude
    of the wave. The images are then stored in Wave folder.
    '''

    A_inverse   = af.lapack.inverse(A_matrix())
    element_LGL = gvar.element_LGL
    delta_t     = gvar.delta_t


    for t_n in trange(0, gvar.time.shape[0] - 1):
        gvar.u[:, :, t_n + 1] = gvar.u[:, :, t_n] + af.blas.matmul(A_inverse,\
                                                                b_vector(t_n))

    print('u calculated!')

    approximate_1_s       = (int(1 / gvar.delta_t) * gvar.delta_t)
    analytical_u_after_1s = np.e ** (-(gvar.element_LGL - gvar.c
                                    * (1 - approximate_1_s)) ** 2 / 0.4 ** 2)

    af.display(analytical_u_after_1s, 10)
    af.display(gvar.u[:, :, int(1 / gvar.delta_t)], 10)
    af.display(gvar.u[:, :, 0], 10)

    subprocess.run(['mkdir', 'results/1D_Wave_images'])

    for t_n in trange(0, gvar.time.shape[0] - 1):
        if t_n % 100 == 0:
            fig = plt.figure()
            x   = gvar.element_LGL
            y   = gvar.u[:, :, t_n]
            
            plt.plot(x, y)
            plt.xlabel('x')
            plt.ylabel('Amplitude')
            plt.title('Time = %f' % (t_n * delta_t))
            fig.savefig('results/1D_Wave_images/%04d' %(t_n / 100) + '.png')
            plt.close('all')

    return
