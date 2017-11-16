#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import arrayfire as af
import numpy as np
import os
from matplotlib import pyplot as pl

from dg_maxwell import wave_equation
from dg_maxwell import msh_parser
from dg_maxwell import wave_equation_2d
from dg_maxwell import isoparam
from dg_maxwell import lagrange
from dg_maxwell import params
from dg_maxwell import utils

def A_matrix():
    '''
    '''

    A_ij = wave_equation_2d.A_matrix(8) / 100

    return A_ij

def volume_integral_vectorized(u):
    '''
    Vectorize, p, q, moddims.
    '''
    dxi_dx   = 10.
    deta_dy  = 10.
    jacobian = 100.

    vol_int_epq = af.np_to_af_array(np.zeros([params.N_LGL ** 2, 100]))

    xi_i  = af.flat(af.transpose(af.tile(params.xi_LGL, 1, params.N_LGL)))
    eta_j = af.tile(params.xi_LGL, params.N_LGL)

    dLp_xi_ij = af.moddims(af.reorder(af.tile(utils.polyval_1d(params.dl_dxi_coeffs,\ 
                xi_i), 1, 1, params.N_LGL), 1, 2, 0), params.N_LGL, 1, params.N_LGL ** 2)
    Lp_xi_ij  = af.moddims(af.reorder(af.tile(utils.polyval_1d(params.lagrange_coeffs,\ 
                xi_i), 1, 1, params.N_LGL), 1, 2, 0), params.N_LGL, 1, params.N_LGL ** 2)

    dLq_eta_ij = af.tile(af.reorder(utils.polyval_1d(params.dl_dxi_coeffs,\
                 params.xi_LGL), 1, 2, 0), 1, 1, params.N_LGL ** 2)
    Lq_eta_ij  = af.tile(af.reorder(utils.polyval_1d(params.lagrange_coeffs,\
                 params.xi_LGL), 1, 2, 0), 1, 1, params.N_LGL ** 2)


    volume_integrand_ij_1 = af.broadcast(utils.multiply,\
                                    Lq_eta_ij * dLp_xi_ij * dxi_dx * params.c_x,\
                                    u) / jacobian

    volume_integrand_ij_2 = af.broadcast(utils.multiply,\
                                    Lp_xi_ij * dLq_eta_ij * deta_dy * params.c_y,\
                                    u) / jacobian

    return vol_int_epq


def volume_integral(u):
    '''
    [TODO] change 'elements'
    '''

    dxi_dx   = 10.
    deta_dy  = 10.
    jacobian = 100.

    vol_int_epq = af.np_to_af_array(np.zeros([params.N_LGL ** 2, 100]))

    xi_i  = af.flat(af.transpose(af.tile(params.xi_LGL, 1, params.N_LGL)))
    eta_j = af.tile(params.xi_LGL, params.N_LGL)
    for p in range (params.N_LGL):
        dLp_xi_ij = af.transpose(utils.polyval_1d(params.dl_dxi_coeffs[p], xi_i))
        Lp_xi_ij  = af.transpose(utils.polyval_1d(params.lagrange_coeffs[p], xi_i))

        for q in range (params.N_LGL):
            index = params.N_LGL * p + q
            dLq_eta_ij = af.transpose(utils.polyval_1d(params.dl_dxi_coeffs[q], eta_j))
            Lq_eta_ij  = af.transpose(utils.polyval_1d(params.lagrange_coeffs[q], eta_j))

            volume_integrand_ij_1 = af.broadcast(utils.multiply,\
                                    Lq_eta_ij * dLp_xi_ij * dxi_dx * params.c_x,\
                                    u) / jacobian
            volume_integrand_ij_2 = af.broadcast(utils.multiply,\
                                    Lp_xi_ij * dLq_eta_ij * deta_dy * params.c_y,\
                                    u) / jacobian
            volume_integrand_interpolate = wave_equation_2d.lag_interpolation_2d(volume_integrand_ij_1\
                                                                                +volume_integrand_ij_2
                                                                                , params.N_LGL)
            volume_integral_e_ij  = af.transpose(utils.integrate_2d_multivar_poly(\
                                           volume_integrand_interpolate, N_quad = 9, scheme = 'gauss'))
            vol_int_epq[index, :] = volume_integral_e_ij

    return vol_int_epq

def lax_friedrichs_flux(u):
    '''
    '''
    u = af.reorder(af.moddims(u, params.N_LGL ** 2, 10, 10), 2, 1, 0)

    diff_u_boundary = af.np_to_af_array(np.zeros([10, 10, params.N_LGL ** 2]))

    u_xi_minus1_boundary_right   = u[:, :, :params.N_LGL]
    u_xi_minus1_boundary_left    = af.shift(u[:, :, -params.N_LGL:], d0=0, d1 = 1)
    u[:, :, :params.N_LGL] = (u_xi_minus1_boundary_right + u_xi_minus1_boundary_left) / 2

    diff_u_boundary[:, :, :params.N_LGL] = (u_xi_minus1_boundary_right - u_xi_minus1_boundary_left)

    u_xi_1_boundary_left  = u[:, :, -params.N_LGL:]
    u_xi_1_boundary_right = af.shift(u[:, :, :params.N_LGL], d0=0, d1=-1)
    u[:, :, :params.N_LGL]     = (u_xi_minus1_boundary_left + u_xi_minus1_boundary_right) / 2

    diff_u_boundary[:, :, -params.N_LGL:] = (u_xi_minus1_boundary_left - u_xi_minus1_boundary_right)

    u_eta_1_boundary_down = u[:, :, params.N_LGL - 1:params.N_LGL ** 2:params.N_LGL]
    u_eta_1_boundary_up   = af.shift(u[:, :, 0:-params.N_LGL + 1:params.N_LGL], d0=1)

    u[:, :, params.N_LGL - 1:params.N_LGL ** 2:params.N_LGL] = (u_eta_1_boundary_up\
                                                              +u_eta_1_boundary_down) / 2

    diff_u_boundary[:, :, params.N_LGL - 1:params.N_LGL ** 2:params.N_LGL] = (u_eta_1_boundary_up\
                                                                             -u_eta_1_boundary_down)


    u_eta_minus1_boundary_down = af.shift(u[:, :, params.N_LGL - 1:params.N_LGL ** 2:params.N_LGL], d0=-1)
    u_eta_minus1_boundary_up   = u[:, :, 0:-params.N_LGL + 1:params.N_LGL]
    u[:, :, 0:-params.N_LGL + 1:params.N_LGL] = (u_eta_minus1_boundary_down\
                                               + u_eta_minus1_boundary_up) / 2
    diff_u_boundary[:, :, params.N_LGL - 1:params.N_LGL ** 2:params.N_LGL] = (u_eta_1_boundary_up\
                                                                             -u_eta_1_boundary_down)


    u_eta_minus1_boundary_down = af.shift(u[:, :, params.N_LGL - 1:params.N_LGL ** 2:params.N_LGL], d0=-1)
    u_eta_minus1_boundary_up   = u[:, :, 0:-params.N_LGL + 1:params.N_LGL]
    u[:, :, 0:-params.N_LGL + 1:params.N_LGL] = (u_eta_minus1_boundary_down\
                                               + u_eta_minus1_boundary_up) / 2

    diff_u_boundary[:, :, 0:-params.N_LGL + 1:params.N_LGL] = (u_eta_minus1_boundary_down\
                                                             - u_eta_minus1_boundary_up)


    u = af.moddims(af.reorder(u, 2, 1, 0), params.N_LGL ** 2, 100)
    diff_u_boundary = af.moddims(af.reorder(diff_u_boundary, 2, 1, 0), params.N_LGL ** 2, 100)
    F_xi_e_ij  = params.c_x * u - params.c_lax * diff_u_boundary
    F_eta_e_ij = params.c_y * u - params.c_lax * diff_u_boundary

    return F_xi_e_ij, F_eta_e_ij


def surface_term(u):
    '''
    '''
    N_LGL = 8
    lagrange_coeffs = params.lagrange_coeffs
    xi_LGL = params.xi_LGL
    eta_LGL = params.xi_LGL
    surface_term_e_ij = af.constant(0., d0 = N_LGL * N_LGL, d1 = 100, dtype = af.Dtype.f64)
    for p in range(params.N_LGL):
        for q in range(params.N_LGL):
            index = p * N_LGL + q
            f_xi_surface_term  = lax_friedrichs_flux(u)[0]
            f_eta_surface_term = lax_friedrichs_flux(u)[1]

            Lp_coeffs = lagrange_coeffs[p]
            Lq_coeffs = lagrange_coeffs[q]
            Lq_eta    = af.transpose(utils.polyval_1d(Lq_coeffs, eta_LGL))
            Lp_xi     = af.transpose(utils.polyval_1d(Lp_coeffs, xi_LGL))
            Lp_1      = utils.polyval_1d(lagrange_coeffs[p], xi_LGL[-1])
            Lq_1      = utils.polyval_1d(lagrange_coeffs[q], eta_LGL[-1])
            Lp_minus1 = utils.polyval_1d(lagrange_coeffs[p], xi_LGL[0])
            Lq_minus1 = utils.polyval_1d(lagrange_coeffs[q], eta_LGL[0])

            # xi = 1 boundary
            Lq_eta_F = af.broadcast(utils.multiply, Lq_eta, f_xi_surface_term[-params.N_LGL:, :])
            Lq_eta_F = af.reorder(Lq_eta_F, 0, 2, 1)

            lag_interpolation_1 = af.reorder(\
                                             af.sum(af.broadcast(utils.multiply, lagrange_coeffs, Lq_eta_F), 0),\
                                             2, 1, 0)

            surface_term_pq_xi_1 = af.sum(Lp_1) * lagrange.integrate(lag_interpolation_1)

            # eta = 1 boundary
            Lp_xi_F = af.broadcast(utils.multiply,\
                                  Lp_xi, f_eta_surface_term[params.N_LGL - 1:params.N_LGL ** 2:params.N_LGL])
            Lp_xi_F = af.reorder(Lp_xi_F, 0, 2, 1)

            lag_interpolation_2 = af.reorder(\
                                  af.sum(af.broadcast(utils.multiply, lagrange_coeffs, Lp_xi_F), 0),\
                                             2, 1, 0)
            surface_term_pq_eta_1 = 1.0 * af.sum(Lq_1) * lagrange.integrate(lag_interpolation_2)

            # xi = -1 boundary
            Lq_eta_F = af.broadcast(utils.multiply,\
                                   Lq_eta, f_xi_surface_term[:params.N_LGL])
            Lq_eta_F = af.reorder(Lq_eta_F, 0, 2, 1)

            lag_interpolation_3 = af.reorder(\
                                  af.sum(af.broadcast(utils.multiply, lagrange_coeffs, Lq_eta_F), 0),\
                                             2, 1, 0)
            surface_term_pq_xi_minus1 = -1.0 * af.sum(Lp_minus1) * lagrange.integrate(lag_interpolation_3)

            # eta = -1 boundary
            Lp_xi_F = af.broadcast(utils.multiply,\
                                  Lp_xi, f_eta_surface_term[0:-params.N_LGL + 1:params.N_LGL])
            Lp_xi_F = af.reorder(Lp_xi_F, 0, 2, 1)

            lag_interpolation_4 = af.reorder(\
                                  af.sum(af.broadcast(utils.multiply, lagrange_coeffs, Lp_xi_F), 0),\
                                             2, 1, 0)
            surface_term_pq_eta_minus1 = -1.0 * af.sum(Lq_minus1) * lagrange.integrate(lag_interpolation_4)

           # print(surface_term_pq_xi_1, surface_term_pq_eta_1, surface_term_pq_eta_minus1, surface_term_pq_xi_minus1)

            surface_term_pq = af.transpose(surface_term_pq_xi_1 + surface_term_pq_eta_1
                                                    + surface_term_pq_xi_minus1 + surface_term_pq_eta_minus1)
            surface_term_e_ij[index] = surface_term_pq

            
    return surface_term_e_ij * 0.1


def b_vector(u):
    '''
    '''
    b = volume_integral(u) - surface_term(u)

    return b

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
    k4 = af.matmul(A_inverse, b_vector(u + k3 * delta_t    ))

    delta_u = delta_t * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return delta_u

