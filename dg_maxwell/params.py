#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import arrayfire as af

backend = 'opencl'
device = 0

af.set_backend(backend)
af.set_device(device)

from dg_maxwell import lagrange
from dg_maxwell import utils
from dg_maxwell import msh_parser
from dg_maxwell import wave_equation
from dg_maxwell import wave_equation_2d

# The domain of the function.
x_nodes    = af.np_to_af_array(np.array([-1., 1.]))

# The number of LGL points into which an element is split.
N_LGL      = 8

# Number of elements the domain is to be divided into.
N_Elements = 10

# The scheme to be used for integration. Values are either
# 'gauss_quadrature' or 'lobatto_quadrature'
scheme     = 'gauss_quadrature'

# The scheme to integrate the volume integral flux
volume_integral_scheme = 'lobatto_quadrature'

# The number quadrature points to be used for integration.
N_quad     = 10

# Wave speed.
c          = 1

# The total time for which the wave is to be evolved by the simulation.
total_time = 2.01

# The c_lax to be used in the Lax-Friedrichs flux.
c_lax      = c

# Array containing the LGL points in xi space.
xi_LGL     = lagrange.LGL_points(N_LGL)


# N_Gauss number of Gauss nodes.
gauss_points               = af.np_to_af_array(lagrange.gauss_nodes(N_quad))

# The Gaussian weights.
gauss_weights              = lagrange.gaussian_weights(N_quad)

# The lobatto nodes to be used for integration.
lobatto_quadrature_nodes   = lagrange.LGL_points(N_quad)

# The lobatto weights to be used for integration.
lobatto_weights_quadrature = lagrange.lobatto_weights\
                                    (N_quad)



# An array containing the coefficients of the lagrange basis polynomials.
lagrange_coeffs            = lagrange.lagrange_polynomial_coeffs(xi_LGL)


lagrange_basis_value = lagrange.lagrange_function_value(lagrange_coeffs)
# While evaluating the volume integral using N_LGL
# lobatto quadrature points, The integration can be vectorized
# and in this case the coefficients of the differential of the
# Lagrange polynomials is required


diff_pow      = (af.flip(af.transpose(af.range(N_LGL - 1) + 1), 1))
dl_dxi_coeffs = (af.broadcast(utils.multiply, lagrange_coeffs[:, :-1], diff_pow))

# Obtaining an array consisting of the LGL points mapped onto the elements.

element_size    = af.sum((x_nodes[1] - x_nodes[0]) / N_Elements)
elements_xi_LGL = af.constant(0, N_Elements, N_LGL)
elements        = utils.linspace(af.sum(x_nodes[0]),
                  af.sum(x_nodes[1] - element_size), N_Elements)

np_element_array   = np.concatenate((af.transpose(elements),
                               af.transpose(elements + element_size)))
element_mesh_nodes = utils.linspace(af.sum(x_nodes[0]),
                                    af.sum(x_nodes[1]), N_Elements + 1)

element_array = af.transpose(af.interop.np_to_af_array(np_element_array))
element_LGL   = wave_equation.mapping_xi_to_x(af.transpose(element_array),\
                                                                   xi_LGL)

# The minimum distance between 2 mapped LGL points.
delta_x = af.min((element_LGL - af.shift(element_LGL, 1, 0))[1:, :])

# dx_dxi for elements of equal size.
dx_dxi  = af.mean(wave_equation.dx_dxi_numerical((element_mesh_nodes[0 : 2]),\
                                   xi_LGL))


# The value of time-step.
delta_t = delta_x / (4 * c)

# Array of timesteps seperated by delta_t.
time    = utils.linspace(0, int(total_time / delta_t) * delta_t,
                                                    int(total_time / delta_t))


# The wave to be advected is either a sin or a Gaussian wave.
# This parameter can take values 'sin' or 'gaussian'.
wave = 'sin'

# Initializing the amplitudes. Change u_init to required initial conditions.
if (wave=='sin'):
    u_init = af.sin(2 * np.pi * element_LGL)

if (wave=='gaussian'):
    u_init = np.e ** (-(element_LGL) ** 2 / 0.4 ** 2)

c_x = 1.

test_array = af.np_to_af_array(np.array(u_init))



# The parameters below are for 2D advection
# -----------------------------------------


########################################################################
#######################2D Wave Equation#################################
########################################################################

c_x = 1.
c_y = 1.

xi_i  = af.flat(af.transpose(af.tile(xi_LGL, 1, N_LGL)))
eta_j = af.tile(xi_LGL, N_LGL)

dLp_xi_ij = af.moddims(af.reorder(af.tile(utils.polyval_1d(dl_dxi_coeffs,
                        xi_i), 1, 1, N_LGL), 1, 2, 0), N_LGL ** 2, 1, N_LGL ** 2)
Lp_xi_ij  = af.moddims(af.reorder(af.tile(utils.polyval_1d(lagrange_coeffs,
                       xi_i), 1, 1, N_LGL), 1, 2, 0), N_LGL ** 2, 1, N_LGL ** 2)

dLq_eta_ij = af.tile(af.reorder(utils.polyval_1d(dl_dxi_coeffs,\
             eta_j), 1, 2, 0), 1, 1, N_LGL)

Lq_eta_ij  = af.tile(af.reorder(utils.polyval_1d(lagrange_coeffs,\
             eta_j), 1, 2, 0), 1, 1, N_LGL)

dLp_xi_ij_Lq_eta_ij = Lq_eta_ij * dLp_xi_ij
dLq_eta_ij_Lp_xi_ij = Lp_xi_ij  * dLq_eta_ij


Li_Lj_coeffs = wave_equation_2d.Li_Lj_coeffs(N_LGL)
courant = 0.1559

delta_y = delta_x

delta_t_2d = courant * delta_x * delta_y / (delta_x * c_x + delta_y * c_y)
