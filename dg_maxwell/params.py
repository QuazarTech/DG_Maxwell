#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import arrayfire as af
af.set_backend('cpu')

from dg_maxwell import lagrange
from dg_maxwell import utils
from dg_maxwell import isoparam

# The domain of the function.
x_nodes    = af.np_to_af_array(np.array([0, 1.]))

# The number of LGL points into which an element is split.
N_LGL      = 8

# Number of elements the domain is to be divided into.
N_Elements = 8

# The scheme to be used for integration. Values are either
# 'gauss_quadrature' or 'lobatto_quadrature'
scheme     = 'gauss_quadrature'

# The scheme to integrate the volume integral flux
volume_integral_scheme = 'lobatto_quadrature'

# The number quadrature points to be used for integration.
N_quad = 8

# Wave speed.
c          = 1

# The total time for which the wave is to be evolved by the simulation.
total_time = 10

# The c_lax to be used in the Lax-Friedrichs flux.
c_lax      = 1

# Array containing the LGL points in xi space.
xi_LGL     = lagrange.LGL_points(N_LGL)

# N_Gauss number of Gauss nodes.
gauss_points  = af.np_to_af_array(lagrange.gauss_nodes(N_quad))

# The Gaussian weights.
gauss_weights = lagrange.gaussian_weights(N_quad)

# The lobatto nodes to be used for integration.
lobatto_quadrature_nodes = lagrange.LGL_points(N_quad)

# The lobatto weights to be used for integration.
lobatto_weights_quadrature = lagrange.lobatto_weights\
                                    (N_quad)

# A list of the Lagrange polynomials in poly1d form.
lagrange_product = lagrange.product_lagrange_poly(xi_LGL)

# An array containing the coefficients of the lagrange basis polynomials.
lagrange_coeffs = af.np_to_af_array(lagrange.lagrange_polynomials(xi_LGL)[1])

# Refer corresponding functions.
lagrange_basis_value = lagrange.lagrange_function_value(lagrange_coeffs)

# A list of the Lagrange polynomials in poly1d form.
lagrange_poly1d_list = lagrange.lagrange_polynomials(xi_LGL)[0]


# list containing the poly1d forms of the differential of Lagrange
# basis polynomials.
differential_lagrange_polynomial = lagrange.differential_lagrange_poly1d()


# While evaluating the volume integral using N_LGL
# lobatto quadrature points, The integration can be vectorized
# and in this case the coefficients of the differential of the
# Lagrange polynomials is required
volume_integrand_8_LGL = np.zeros(([N_LGL, N_LGL - 1]))

for i in range(N_LGL):
    volume_integrand_8_LGL[i] = (differential_lagrange_polynomial[i]).c

volume_integrand_8_LGL= af.np_to_af_array(volume_integrand_8_LGL)

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
element_LGL   = isoparam.isoparam_1D(af.transpose(element_array),
                                              xi_LGL)

# The minimum distance between 2 mapped LGL points.
delta_x = af.min((element_LGL - af.shift(element_LGL, 1, 0))[1:, :])

# The value of time-step.
delta_t = delta_x / (20 * c)

# Array of timesteps seperated by delta_t.
time    = utils.linspace(0, int(total_time / delta_t) * delta_t,
                                                    int(total_time / delta_t))

# Initializing the amplitudes. Change u_init to required initial conditions.
u_init     = af.sin(2 * np.pi * element_LGL)
u          = af.constant(0, N_LGL, N_Elements, time.shape[0],\
                                 dtype = af.Dtype.f64)
u[:, :, 0] = u_init
