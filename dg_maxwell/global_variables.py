#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import arrayfire as af

from dg_maxwell import params
from dg_maxwell import lagrange
from dg_maxwell import utils
from dg_maxwell import wave_equation

af.set_backend(params.backend)
af.set_device(params.device)

class global_variables:
    '''
    '''
    
    def __init__(self, N_LGL, N_quad, x_nodes, N_elements, c):
        '''
        '''
        self.xi_LGL = lagrange.LGL_points(N_LGL)
        
        # N_Gauss number of Gauss nodes.
        self.gauss_points = af.np_to_af_array(lagrange.gauss_nodes(N_quad))

        # The Gaussian weights.
        self.gauss_weights = lagrange.gaussian_weights(N_quad)

        # The lobatto nodes to be used for integration.
        self.lobatto_quadrature_nodes = lagrange.LGL_points(N_quad)

        # The lobatto weights to be used for integration.
        self.lobatto_weights_quadrature = lagrange.lobatto_weights(N_quad)

        # An array containing the coefficients of the lagrange basis polynomials.
        self.lagrange_coeffs = lagrange.lagrange_polynomial_coeffs(self.xi_LGL)
        
        self.lagrange_basis_value = lagrange.lagrange_function_value(self.lagrange_coeffs)

        self.diff_pow = af.flip(af.transpose(af.range(N_LGL - 1) + 1), 1)
        self.dl_dxi_coeffs = af.broadcast(utils.multiply, self.lagrange_coeffs[:, :-1],
                                          self.diff_pow)

        self.element_size    = af.sum((x_nodes[1] - x_nodes[0]) / N_elements)
        self.elements_xi_LGL = af.constant(0, N_elements, N_LGL)
        self.elements        = utils.linspace(af.sum(x_nodes[0]),
                                              af.sum(x_nodes[1]
                                                     - self.element_size),
                                              N_elements)

        self.np_element_array = np.concatenate(
            af.transpose(self.elements),
            af.transpose(self.elements
                         + self.element_size))
        self.element_mesh_nodes = utils.linspace(af.sum(x_nodes[0]),
                                                 af.sum(x_nodes[1]),
                                                 N_elements + 1)

        self.element_array = af.transpose(
            af.interop.np_to_af_array(self.np_element_array))
        self.element_LGL   = wave_equation.mapping_xi_to_x(
            af.transpose(self.element_array), self.xi_LGL)

        # The minimum distance between 2 mapped LGL points.
        self.delta_x = af.min((self.element_LGL - af.shift(self.element_LGL,
                                                           1, 0))[1:, :])

        # dx_dxi for elements of equal size.
        self.dx_dxi  = af.mean(wave_equation.dx_dxi_numerical(
            self.element_mesh_nodes[0 : 2],
            self.xi_LGL))

        # The value of time-step.
        self.delta_t = self.delta_x / (4 * c)

        # Array of timesteps seperated by delta_t.
        self.time = None

        # The wave to be advected is either a sin or a Gaussian wave.
        # This parameter can take values 'sin' or 'gaussian'.
        self.wave = None

        # Initializing the amplitudes. Change u_init to required initial conditions.
        if (self.wave=='sin'):
            self.u_init = af.sin(2 * np.pi * element_LGL)

        if (wave=='gaussian'):
            self.u_init = np.e ** (-(element_LGL) ** 2 / 0.4 ** 2)

        self.test_array = af.np_to_af_array(np.array(u_init))



        # The parameters below are for 2D advection
        # -----------------------------------------


        ########################################################################
        #######################2D Wave Equation#################################
        ########################################################################


        self.xi_i  = None
        self.eta_j = None

        self.dLp_xi_ij = None
        self.Lp_xi_ij  = None

        self.dLq_eta_ij = None

        self.Lq_eta_ij  = None

        self.dLp_xi_ij_Lq_eta_ij = None
        self.dLq_eta_ij_Lp_xi_ij = None

        self.Li_Lj_coeffs = None

        self.delta_y = None

        self.delta_t_2d = None

        self.c_lax_2d_x = None
        self.c_lax_2d_y = None

        self.nodes = None
        self.elements = None

        self.x_e_ij = None
        self.y_e_ij = None
        
        self.x_e_ij = None
        self.y_e_ij = None

        self.u_e_ij = None

        # Array of timesteps seperated by delta_t.
        self.time_2d = None

        return
    