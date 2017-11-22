import arrayfire as af
import numpy as np
import os
import numpy as np
import arrayfire as af

from matplotlib import pyplot as pl
from tqdm import trange

from dg_maxwell import wave_equation
from dg_maxwell import msh_parser
from dg_maxwell import wave_equation_2d
from dg_maxwell.tests import test_waveEqn
from dg_maxwell import isoparam
from dg_maxwell import lagrange
from dg_maxwell import params
from dg_maxwell import advection_2d
from dg_maxwell import utils

af.set_backend(params.backend)

print(af.info())
#print(af.mean(af.abs(advection_2d.u_analytical(0) - params.u_e_ij)))


def change_parameters(LGL, Elements=10, wave='sin'):
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
    #params.lagrange_product = lagrange.product_lagrange_poly(params.xi_LGL)

    # An array containing the coefficients of the lagrange basis polynomials.
    params.lagrange_coeffs  = lagrange.lagrange_polynomial_coeffs(params.xi_LGL)

    # Refer corresponding functions.
    params.lagrange_basis_value = lagrange.lagrange_function_value\
                                           (params.lagrange_coeffs)


    # While evaluating the volume integral using N_LGL
    # lobatto quadrature points, The integration can be vectorized
    # and in this case the coefficients of the differential of the
    # Lagrange polynomials is required
    params.diff_pow      = (af.flip(af.transpose(af.range(params.N_LGL - 1) + 1), 1))
    params.dl_dxi_coeffs = (af.broadcast(utils.multiply, params.lagrange_coeffs[:, :-1], params.diff_pow))



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
    params.element_LGL   = wave_equation.mapping_xi_to_x(af.transpose\
                                          (params.element_array), params.xi_LGL)

    # The minimum distance between 2 mapped LGL points.
    params.delta_x = af.min((params.element_LGL - af.shift(params.element_LGL, 1, 0))[1:, :])

    # dx_dxi for elements of equal size.
    params. dx_dxi = af.mean(wave_equation.dx_dxi_numerical((params.element_mesh_nodes[0 : 2]),\
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

    # The parameters below are for 2D advection
    # -----------------------------------------
    
    
    ########################################################################
    #######################2D Wave Equation#################################
    ########################################################################
    
    params.c_x = 1.
    params.c_y = 1.
    
    params.xi_i  = af.flat(af.transpose(af.tile(params.xi_LGL, 1, params.N_LGL)))
    params.eta_j = af.tile(params.xi_LGL, params.N_LGL)
    
    params.dLp_xi_ij = af.moddims(af.reorder(af.tile(utils.polyval_1d(params.dl_dxi_coeffs,
                            params.xi_i), 1, 1, params.N_LGL), 1, 2, 0), params.N_LGL ** 2, 1, params.N_LGL ** 2)
    params.Lp_xi_ij  = af.moddims(af.reorder(af.tile(utils.polyval_1d(params.lagrange_coeffs,
                           params.xi_i), 1, 1, params.N_LGL), 1, 2, 0), params.N_LGL ** 2, 1, params.N_LGL ** 2)
    
    params.dLq_eta_ij = af.tile(af.reorder(utils.polyval_1d(params.dl_dxi_coeffs,\
                 params.eta_j), 1, 2, 0), 1, 1, params.N_LGL)
    
    params.Lq_eta_ij  = af.tile(af.reorder(utils.polyval_1d(params.lagrange_coeffs,\
                 params.eta_j), 1, 2, 0), 1, 1, params.N_LGL)
    
    params.dLp_xi_ij_Lq_eta_ij = params.Lq_eta_ij * params.dLp_xi_ij
    params.dLq_eta_ij_Lp_xi_ij = params.Lp_xi_ij  * params.dLq_eta_ij
    
    
    params.Li_Lj_coeffs = wave_equation_2d.Li_Lj_coeffs(params.N_LGL)
    params.courant = 0.1
    
    params.delta_y = params.delta_x
    
    params.delta_t_2d = params.courant * params.delta_x * params.delta_y / (params.delta_x * params.c_x + params.delta_y * params.c_y)

    params.c_lax_2d_x = params.c_x
    params.c_lax_2d_y = params.c_y

    params.nodes, params.elements = msh_parser.read_order_2_msh('square_10_10.msh')
    
    params.x_e_ij  = af.np_to_af_array(np.zeros([params.N_LGL * params.N_LGL, len(params.elements)]))
    params.y_e_ij  = af.np_to_af_array(np.zeros([params.N_LGL * params.N_LGL, len(params.elements)]))
    
    for element_tag, element in enumerate(params.elements):
        params.x_e_ij[:, element_tag] = isoparam.isoparam_x_2D(params.nodes[element, 0], params.xi_i, params.eta_j)
        params.y_e_ij[:, element_tag] = isoparam.isoparam_y_2D(params.nodes[element, 1], params.xi_i, params.eta_j)
    
    #u_e_ij = np.e ** (-(x_e_ij ** 2 + y_e_ij ** 2)/(0.4 ** 2))
    params.u_e_ij = af.sin(params.x_e_ij * 2 * np.pi + params.y_e_ij * 4 * np.pi)
    
    params.total_time_2d = 0.1
    
    # Array of timesteps seperated by delta_t.
    params.time_2d  = utils.linspace(0, int(params.total_time_2d / params.delta_t_2d) * params.delta_t_2d,
                                                    int(params.total_time_2d / params.delta_t_2d))

    return

change_parameters(9)
print(advection_2d.time_evolution())

#L1_norm = np.zeros([5])
#for LGL in range(3, 8):
#    print(LGL)
#    change_parameters(LGL)
#    L1_norm[LGL - 3] = (advection_2d.time_evolution())
#    print(L1_norm[LGL - 3])
#
#print(L1_norm)

L1_norm = np.array([8.20284941e-02, 1.05582246e-02, 9.12125969e-04, 1.26001632e-04, 8.97007162e-06, 1.0576058855881385e-06])
LGL = (np.arange(6) + 3).astype(float)
normalization = 8.20284941e-02 / (3 ** (-3 * 0.85))
pl.loglog(LGL, L1_norm, marker='o', label='L1 norm')
pl.loglog(LGL, normalization * LGL ** (-0.85 * LGL), color='black', linestyle='--', label='$N_{LGL}^{-0.85 N_{LGL}}$')
pl.title('L1 norm v/s $N_{LGL}$')
pl.legend(loc='best')
pl.xlabel('$N_{LGL}$ points')
pl.ylabel('L1 norm of error')
pl.show()
