#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import arrayfire as af
import os
import h5py


from dg_maxwell import params
from dg_maxwell import isoparam
from dg_maxwell import msh_parser
from dg_maxwell import advection_2d
from dg_maxwell import lagrange
from dg_maxwell import utils

from tqdm import trange

import os
import sys
import csv
sys.path.insert(0, os.path.abspath('../'))

af.set_backend(params.backend)
af.set_device(params.device)

def dx_dxi(x_nodes, xi, eta):
    '''
    Computes the derivative :math:`\\frac{\\partial x}{\\partial \\xi}`.
    The derivative is obtained by finding the derivative of the analytical
    function :math:`x \\equiv x(\\xi, \\eta)`.

    The derivation of the analytical form of
    :math:`\\frac{\\partial x}{\\partial \\xi}` is given in this `worksheet`_.

    .. _worksheet: https://goo.gl/ffCJvn

    Parameters
    ----------
    x_nodes : np.ndarray [8]
              :math:`x` nodes.

    xi      : af.Array
              :math:`\\xi` coordinates for which
              :math:`\\frac{\\partial x}{\\partial \\xi}` has to be found.
              This could be a single number or a meshgrid.

    eta     : af.Array
              :math:`\\eta` coordinate for which
              :math:`\\frac{\\partial x}{\\partial \\xi}` has to be found.
              This could be a single number or a meshgrid.

    Returns
    -------
    dx_dxi : af.Array
             :math:`\\frac{\\partial x}{\\partial \\xi}` calculated at
             :math:`(\\xi, \\eta)` coordinate.
    '''

    dN_0_dxi = -0.25*eta**2 + (0.5*eta + 0.5)*xi - 0.25*eta
    dN_1_dxi = 0.5*eta**2 - 0.5
    dN_2_dxi = -0.25*eta**2 + (-0.5*eta + 0.5)*xi + 0.25*eta
    dN_3_dxi = (eta - 1.0)*xi
    dN_4_dxi = 0.25*eta**2 + (-0.5*eta + 0.5)*xi - 0.25*eta
    dN_5_dxi = -0.5*eta**2 + 0.5
    dN_6_dxi = 0.25*eta**2 + (0.5*eta + 0.5)*xi + 0.25*eta
    dN_7_dxi = (-1.0*eta - 1.0)*xi

    dx_dxi = dN_0_dxi * x_nodes[0] \
           + dN_1_dxi * x_nodes[1] \
           + dN_2_dxi * x_nodes[2] \
           + dN_3_dxi * x_nodes[3] \
           + dN_4_dxi * x_nodes[4] \
           + dN_5_dxi * x_nodes[5] \
           + dN_6_dxi * x_nodes[6] \
           + dN_7_dxi * x_nodes[7]

    return dx_dxi


def dx_deta(x_nodes, xi, eta):
    '''
    Computes the derivative :math:`\\frac{\\partial x}{\\partial \\eta}`.
    The derivative is obtained by finding the derivative of the analytical
    function :math:`x \\equiv x(\\xi, \\eta)`.

    Parameters
    ----------
    y_nodes : np.ndarray [8]
              :math:`y` nodes.

    xi      : af.Array
              :math:`\\xi` coordinate for which
              :math:`\\frac{\\partial x}{\\partial \\eta}` has to be found.

    eta     : af.Array
              :math:`\\eta` coordinate for which
              :math:`\\frac{\\partial x}{\\partial \\eta}` has to be found.

    Returns
    -------
    dx_deta : af.Array
              :math:`\\frac{\\partial x}{\\partial \\eta}` calculated at
              :math:`(\\xi, \\eta)` coordinate.
    '''

    dN_0_deta = -(eta - xi - 1) * (0.25 * xi - 0.25) \
                - (eta + 1) * (0.25 * xi - 0.25)
    dN_1_deta = -2 * eta * (-0.5 * xi + 0.5)
    dN_2_deta = -(eta + xi + 1) * (0.25 * xi - 0.25) \
                - (eta - 1) * (0.25 * xi - 0.25)
    dN_3_deta = 0.5 * xi**2 - 0.5
    dN_4_deta = -(eta - xi + 1) * (-0.25 * xi - 0.25) \
                - (eta - 1) * (-0.25 * xi - 0.25)
    dN_5_deta = -2 * eta * (0.5 * xi + 0.5)
    dN_6_deta = -(eta + xi - 1) * (-0.25 * xi - 0.25) \
                - (eta + 1) * (-0.25 * xi - 0.25)
    dN_7_deta = -0.5 * xi**2 + 0.5

    dx_deta = dN_0_deta * x_nodes[0] \
            + dN_1_deta * x_nodes[1] \
            + dN_2_deta * x_nodes[2] \
            + dN_3_deta * x_nodes[3] \
            + dN_4_deta * x_nodes[4] \
            + dN_5_deta * x_nodes[5] \
            + dN_6_deta * x_nodes[6] \
            + dN_7_deta * x_nodes[7]

    return dx_deta


def dy_dxi(y_nodes, xi, eta):
    '''
    Computes the derivative :math:`\\frac{\\partial y}{\\partial \\xi}`.
    The derivative is obtained by finding the derivative of the analytical
    function :math:`y \\equiv y(\\xi, \\eta)`.

    Parameters
    ----------
    y_nodes : np.ndarray [8]
              :math:`y` nodes.

    xi      : af.Array
              :math:`\\xi` coordinate for which
              :math:`\\frac{\\partial y}{\\partial \\xi}` has to be found.

    eta     : af.Array
              :math:`\\eta` coordinate for which
              :math:`\\frac{\\partial y}{\\partial \\xi}` has to be found.

    Returns
    -------
    af.Array
        :math:`\\frac{\\partial y}{\\partial \\xi}` calculated at
        :math:`(\\xi, \\eta)` coordinate.
    '''
    return dx_dxi(y_nodes, xi, eta)


def dy_deta(y_nodes, xi, eta):
    '''
    Computes the derivative :math:`\\frac{\\partial y}{\\partial \\eta}`.
    The derivative is obtained by finding the derivative of the analytical
    function :math:`y \\equiv y(\\xi, \\eta)`.

    Parameters
    ----------
    y_nodes : np.ndarray [8]
              :math:`y` nodes.

    xi      : af.Array
              :math:`\\xi` coordinate for which
              :math:`\\frac{\\partial y}{\\partial \\eta}` has to be found.

    eta     : af.Array
              :math:`\\eta` coordinate for which
              :math:`\\frac{\\partial y}{\\partial \\eta}` has to be found.

    Returns
    -------
    af.Array
        :math:`\\frac{\\partial y}{\\partial \\eta}` calculated at
        :math:`(\\xi, \\eta)` coordinate.
    '''
    return dx_deta(y_nodes, xi, eta)


def jacobian(x_nodes, y_nodes, xi, eta):
    '''
    Calculates the jocobian for the corrdinate transformation from
    :math:`xy` space to :math:`\\xi \\eta` space.

    Parameters
    ----------
    x_nodes : np.ndarray [8]
              :math:`x` nodes.

    y_nodes : np.ndarray [8]
              :math:`y` nodes.

    xi      : af.Array
              :math:`\\xi` coordinate at which
              Jacobian has to be found.

    eta     : af.Array
              :math:`\\eta` coordinate at which
              Jacobian has to be found.

    Returns
    -------
    float
        Returns the Jacobian calculated using this formula.

        .. math::   J\\Big(\\frac{x, y}{\\xi, \\eta}\Big) =
                    \\frac{\\partial x}{\\partial \\xi}  \
                    \\frac{\\partial y}{\\partial \\eta} \
                    - \\frac{\\partial x}{\\partial \\eta} \
                    \\frac{\\partial y}{\\partial \\xi}
    '''

    dx_dxi_  = dx_dxi (x_nodes, xi, eta)
    dy_deta_ = dy_deta (y_nodes, xi, eta)
    dx_deta_ = dx_deta (x_nodes, xi, eta)
    dy_dxi_  = dy_dxi (y_nodes, xi, eta)
    
    return (dx_dxi_ * dy_deta_) - (dx_deta_ * dy_dxi_)

def dxi_dx(x_nodes, y_nodes, xi, eta):
    '''
    '''
    dy_deta_ = dy_deta(y_nodes, xi, eta)
    
    return dy_deta_ / jacobian(x_nodes, y_nodes, xi, eta)

def dxi_dy(x_nodes, y_nodes, xi, eta):
    '''
    '''
    dx_deta_ = dx_deta(x_nodes, xi, eta)
    
    return -dx_deta_ / jacobian(x_nodes, y_nodes, xi, eta)

def deta_dx(x_nodes, y_nodes, xi, eta):
    '''
    '''
    dy_dxi_ = dy_dxi(y_nodes, xi, eta)
    
    return -dy_dxi_ / jacobian(x_nodes, y_nodes, xi, eta)

def deta_dy (x_nodes, y_nodes, xi, eta):
    '''
    '''
    dx_dxi_ = dx_dxi(x_nodes, xi, eta)
    
    return dx_dxi_ / jacobian(x_nodes, y_nodes, xi, eta)

# Functions which perform the same function as the above ones
# but can compute for different elements.

def trial_dx_dxi(x_nodes, xi, eta):
    '''
    '''

    dN_0_dxi = -0.25*eta**2 + (0.5*eta + 0.5)*xi - 0.25*eta
    dN_1_dxi = 0.5*eta**2 - 0.5
    dN_2_dxi = -0.25*eta**2 + (-0.5*eta + 0.5)*xi + 0.25*eta
    dN_3_dxi = (eta - 1.0)*xi
    dN_4_dxi = 0.25*eta**2 + (-0.5*eta + 0.5)*xi - 0.25*eta
    dN_5_dxi = -0.5*eta**2 + 0.5
    dN_6_dxi = 0.25*eta**2 + (0.5*eta + 0.5)*xi + 0.25*eta
    dN_7_dxi = (-1.0*eta - 1.0)*xi

    dx_dxi = af.broadcast(utils.multiply, dN_0_dxi, x_nodes[0]) \
           + af.broadcast(utils.multiply, dN_1_dxi, x_nodes[1]) \
           + af.broadcast(utils.multiply, dN_2_dxi, x_nodes[2]) \
           + af.broadcast(utils.multiply, dN_3_dxi, x_nodes[3]) \
           + af.broadcast(utils.multiply, dN_4_dxi, x_nodes[4]) \
           + af.broadcast(utils.multiply, dN_5_dxi, x_nodes[5]) \
           + af.broadcast(utils.multiply, dN_6_dxi, x_nodes[6]) \
           + af.broadcast(utils.multiply, dN_7_dxi, x_nodes[7])

    return dx_dxi

def trial_dx_deta(x_nodes, xi, eta):
    '''
    '''

    dN_0_deta = -(eta - xi - 1) * (0.25 * xi - 0.25) \
                - (eta + 1) * (0.25 * xi - 0.25)
    dN_1_deta = -2 * eta * (-0.5 * xi + 0.5)
    dN_2_deta = -(eta + xi + 1) * (0.25 * xi - 0.25) \
                - (eta - 1) * (0.25 * xi - 0.25)
    dN_3_deta = 0.5 * xi**2 - 0.5
    dN_4_deta = -(eta - xi + 1) * (-0.25 * xi - 0.25) \
                - (eta - 1) * (-0.25 * xi - 0.25)
    dN_5_deta = -2 * eta * (0.5 * xi + 0.5)
    dN_6_deta = -(eta + xi - 1) * (-0.25 * xi - 0.25) \
                - (eta + 1) * (-0.25 * xi - 0.25)
    dN_7_deta = -0.5 * xi**2 + 0.5

    dx_deta = af.broadcast(utils.multiply, dN_0_deta, x_nodes[0, :, :]) \
            + af.broadcast(utils.multiply, dN_1_deta, x_nodes[1, :, :]) \
            + af.broadcast(utils.multiply, dN_2_deta, x_nodes[2, :, :]) \
            + af.broadcast(utils.multiply, dN_3_deta, x_nodes[3, :, :]) \
            + af.broadcast(utils.multiply, dN_4_deta, x_nodes[4, :, :]) \
            + af.broadcast(utils.multiply, dN_5_deta, x_nodes[5, :, :]) \
            + af.broadcast(utils.multiply, dN_6_deta, x_nodes[6, :, :]) \
            + af.broadcast(utils.multiply, dN_7_deta, x_nodes[7, :, :])

    return dx_deta



def trial_dy_dxi(y_nodes, xi, eta):
    '''
    '''
    return trial_dx_dxi(y_nodes, xi, eta)


def trial_dy_deta(y_nodes, xi, eta):
    '''
    '''
    return trial_dx_deta(y_nodes, xi, eta)


def trial_jacobian(x_nodes, y_nodes, xi, eta):
    '''
    '''

    dx_dxi_  = trial_dx_dxi (x_nodes, xi, eta)
    dy_deta_ = trial_dy_deta (y_nodes, xi, eta)
    dx_deta_ = trial_dx_deta (x_nodes, xi, eta)
    dy_dxi_  = trial_dy_dxi (y_nodes, xi, eta)
    
    return (dx_dxi_ * dy_deta_) - (dx_deta_ * dy_dxi_)

def trial_dxi_dx(x_nodes, y_nodes, xi, eta):
    '''
    '''
    dy_deta_ = trial_dy_deta(y_nodes, xi, eta)
    
    return dy_deta_ / trial_jacobian(x_nodes, y_nodes, xi, eta)

def trial_dxi_dy(x_nodes, y_nodes, xi, eta):
    '''
    '''
    dx_deta_ = trial_dx_deta(x_nodes, xi, eta)
    
    return -dx_deta_ / trial_jacobian(x_nodes, y_nodes, xi, eta)

def trial_deta_dx(x_nodes, y_nodes, xi, eta):
    '''
    '''
    dy_dxi_ = trial_dy_dxi(y_nodes, xi, eta)
    
    return -dy_dxi_ / trial_jacobian(x_nodes, y_nodes, xi, eta)

def trial_deta_dy (x_nodes, y_nodes, xi, eta):
    '''
    '''
    dx_dxi_ = trial_dx_dxi(x_nodes, xi, eta)
    
    return dx_dxi_ / trial_jacobian(x_nodes, y_nodes, xi, eta)

##############

def A_matrix(N_LGL, advec_var):
    '''
    Calculates the tensor product for the given ``params.N_LGL``.
    A tensor product element is given by:

    .. math:: [A^{pq}_{ij}] = \\iint L_p(\\xi) L_q(\\eta) \\
                                     L_i(\\xi) L_j(\\eta) d\\xi d\\eta

    This function finds :math:`L_p(\\xi) L_i(\\xi)` and
    :math:`L_q(\\eta) L_j(\\eta)` and passes it to the ``integrate_2d``
    function.

    Returns
    -------
    A : af.Array [N_LGL^2 N_LGL^2 1 1]
        The tensor product.
    '''
    xi_LGL = lagrange.LGL_points(N_LGL)
    lagrange_coeffs = af.np_to_af_array(lagrange.lagrange_polynomials(xi_LGL)[1])

    xi_LGL  = lagrange.LGL_points(N_LGL)
    eta_LGL = lagrange.LGL_points(N_LGL)

    _, Lp_xi  = lagrange.lagrange_polynomials(xi_LGL)
    _, Lq_eta = lagrange.lagrange_polynomials(eta_LGL)
    Lp_xi = af.np_to_af_array(Lp_xi)
    Lq_eta = af.np_to_af_array(Lq_eta)
    Li_xi = Lp_xi.copy()
    Lj_eta = Lq_eta.copy()

    Lp_xi_tp = af.reorder(Lp_xi, d0 = 2, d1 = 0, d2 = 1)
    Lp_xi_tp = af.tile(Lp_xi_tp, d0 = N_LGL * N_LGL * N_LGL)
    Lp_xi_tp = af.moddims(Lp_xi_tp, d0 = N_LGL * N_LGL * N_LGL * N_LGL, d1 = 1, d2 = N_LGL)
    Lp_xi_tp = af.reorder(Lp_xi_tp, d0 = 0, d1 = 2, d2 = 1)

    Lq_eta_tp = af.reorder(Lq_eta, d0 = 0, d1 = 2, d2 = 1)
    Lq_eta_tp = af.tile(Lq_eta_tp, d0 = N_LGL, d1 = N_LGL * N_LGL)
    Lq_eta_tp = af.moddims(af.transpose(Lq_eta_tp), d0 = N_LGL * N_LGL * N_LGL * N_LGL, d1 = 1, d2 = N_LGL)
    Lq_eta_tp = af.reorder(Lq_eta_tp, d0 = 0, d1 = 2, d2 = 1)

    Li_xi_tp = af.reorder(Li_xi, d0 = 2, d1 = 0, d2 = 1)
    Li_xi_tp = af.tile(Li_xi_tp, d0 = N_LGL)
    Li_xi_tp = af.moddims(Li_xi_tp, d0 = N_LGL * N_LGL, d1 = 1, d2 = N_LGL)
    Li_xi_tp = af.reorder(Li_xi_tp, d0 = 0, d1 = 2, d2 = 1)
    Li_xi_tp = af.tile(Li_xi_tp, d0 = N_LGL * N_LGL)

    Lj_eta_tp = af.reorder(Lj_eta, d0 = 0, d1 = 2, d2 = 1)
    Lj_eta_tp = af.tile(Lj_eta_tp, d0 = N_LGL)
    Lj_eta_tp = af.reorder(Lj_eta_tp, d0 = 0, d1 = 2, d2 = 1)
    Lj_eta_tp = af.tile(Lj_eta_tp, d0 = N_LGL * N_LGL)

    Lp_Li_tp = utils.poly1d_product(Lp_xi_tp, Li_xi_tp)
    Lq_Lj_tp = utils.poly1d_product(Lq_eta_tp, Lj_eta_tp)

    Lp_Li_Lq_Lj_tp = utils.polynomial_product_coeffs(af.reorder(Lp_Li_tp,
                                                                d0 = 1,
                                                                d1 = 2,
                                                                d2 = 0),
                                                     af.reorder(Lq_Lj_tp,
                                                                d0 = 1,
                                                                d1 = 2,
                                                                d2 = 0))

    A = utils.integrate_2d_multivar_poly(Lp_Li_Lq_Lj_tp, params.N_quad,
                                         'gauss', advec_var)

    A = af.moddims(A, d0 = N_LGL * N_LGL, d1 = N_LGL * N_LGL)

    return A

def A_matrix_xi_eta(N_LGL, advec_var):
    '''
    '''
    A_xi_eta = A_matrix(N_LGL, advec_var) * np.mean(advec_var.sqrt_det_g)
    return A_xi_eta



def F_x(u):
    '''
    '''
    return params.c_x * u


def F_y(u):
    '''
    '''
    return params.c_y * u



def g_dd(x_nodes, y_nodes, xi, eta):
    '''
    '''
    ans00  =   (dx_dxi(x_nodes, xi, eta))**2 \
             + (dy_dxi(y_nodes, xi, eta))**2
    ans11  =   (dx_deta(x_nodes, xi, eta))**2 \
             + (dy_deta(y_nodes, xi, eta))**2
    
    ans01  =  (dx_dxi(x_nodes, xi, eta))  \
            * (dx_deta(x_nodes, xi, eta)) \
            + (dy_dxi(y_nodes, xi, eta))  \
            * (dy_deta(y_nodes, xi, eta))
    
    ans =  [[ans00, ans01],
            [ans01, ans11]
           ]
    
    return np.array(ans)


def g_uu(x_nodes, y_nodes, xi, eta):
    gCov = g_dd(x_nodes, y_nodes, xi, eta)
    
    
    a = gCov[0][0]
    b = gCov[0][1]
    c = gCov[1][0]
    d = gCov[1][1]

    det = (a*d - b*c)

    ans = [[d / det, -b / det],
           [-c / det, a / det]]

    return ans


def sqrt_det_g(x_nodes, y_nodes, xi, eta):
    '''
    '''
    gCov = g_dd(x_nodes, y_nodes, xi, eta)
    
    a = gCov[0][0]
    b = gCov[0][1]
    c = gCov[1][0]
    d = gCov[1][1]
    
    return (a*d - b*c)**0.5

# Trial functions which compute the metric tensor for multiple elements.

def trial_g_dd(x_nodes, y_nodes, xi, eta):
    '''
    '''
    ans00  =   (trial_dx_dxi(x_nodes, xi, eta))**2 \
             + (trial_dy_dxi(y_nodes, xi, eta))**2
    ans11  =   (trial_dx_deta(x_nodes, xi, eta))**2 \
             + (trial_dy_deta(y_nodes, xi, eta))**2
    
    ans01  =  (trial_dx_dxi(x_nodes, xi, eta))  \
            * (trial_dx_deta(x_nodes, xi, eta)) \
            + (trial_dy_dxi(y_nodes, xi, eta))  \
            * (trial_dy_deta(y_nodes, xi, eta))


    ans =  [[ans00, ans01],
            [ans01, ans11]
           ]

    return (ans)


def trial_g_uu(x_nodes, y_nodes, xi, eta):
    gCov = trial_g_dd(x_nodes, y_nodes, xi, eta)


    a = gCov[0][0]
    b = gCov[0][1]
    c = gCov[1][0]
    d = gCov[1][1]

    det = (a*d - b*c)

    ans = [[d / det, -b / det],
           [-c / det, a / det]]

    return ans


def trial_sqrt_det_g(x_nodes, y_nodes, xi, eta):
    '''
    '''
    gCov = trial_g_dd(x_nodes, y_nodes, xi, eta)
    
    a = gCov[0][0]
    b = gCov[0][1]
    c = gCov[1][0]
    d = gCov[1][1]

    return (a*d - b*c)**0.5
    
############## 

def F_xi(u, gv):
    '''
    '''
    nodes    = gv.nodes
    elements = gv.elements

    xi_LGL = gv.xi_LGL
    xi_i   = gv.xi_i
    eta_j  = gv.eta_j

    dxi_by_dx = af.reorder(trial_dxi_dx(gv.elements_nodes[:, 0, :], gv.elements_nodes[:, 1, :], xi_i, eta_j), 0, 2, 1)
    dxi_by_dy = af.reorder(trial_dxi_dy(gv.elements_nodes[:, 0, :], gv.elements_nodes[:, 1, :], xi_i, eta_j), 0, 2, 1)
    F_xi_u = F_x(u) * dxi_by_dx + F_y(u) * dxi_by_dy

    return F_xi_u


def F_eta(u, gv):
    '''
    '''
    nodes    = gv.nodes
    elements = gv.elements

    xi_LGL = gv.xi_LGL
    xi_i   = gv.xi_i
    eta_j  = gv.eta_j


    deta_by_dx = af.reorder(trial_deta_dx(gv.elements_nodes[:, 0, :], gv.elements_nodes[:, 1, :], xi_i, eta_j), 0, 2, 1)
    deta_by_dy = af.reorder(trial_deta_dy(gv.elements_nodes[:, 0, :], gv.elements_nodes[:, 1, :], xi_i, eta_j), 0, 2, 1)

    F_eta_u = F_x(u) * deta_by_dx + F_y(u) * deta_by_dy

    return F_eta_u


def Li_Lj_coeffs(N_LGL):
    '''
    '''
    xi_LGL = lagrange.LGL_points(N_LGL)
    lagrange_coeffs = af.np_to_af_array(lagrange.lagrange_polynomials(xi_LGL)[1])

    Li_xi  = af.moddims(af.tile(af.reorder(lagrange_coeffs, 1, 2, 0),
                                1, N_LGL),
                        N_LGL, 1, N_LGL ** 2)
    
    Lj_eta = af.tile(af.reorder(lagrange_coeffs, 1, 2, 0), 1, 1, N_LGL)

    Li_Lj_coeffs = utils.polynomial_product_coeffs(Li_xi, Lj_eta)

    return Li_Lj_coeffs


def lag_interpolation_2d(u_e_ij, Li_Lj_coeffs):
    '''
    Does the lagrange interpolation of a function.
    
    Parameters
    ----------
    
    u_e_ij : af.Array [N_LGL^2 N_elements 1 1]
             Value of the function calculated at the :math:`(\\xi_i, \\eta_j)`
             points in this form
             
             .. math:: \\xi_i = [\\xi_0, \\xi_0, ..., \\xi_0, \\xi_1, \\
                       ... ..., \\xi_N]
             .. math:: \\eta_j = [\\eta_0, \\eta_1, ..., \\eta_N, \\
                       \\eta_0, ... ..., \\eta_N]

    N_LGL : int
            Number of LGL points

    Returns
    -------
    interpolated_f : af.Array [N_LGL N_LGL N_elements 1]
                     Interpolation polynomials for ``N_elements`` elements.
    '''
    Li_xi_Lj_eta_coeffs = af.tile(Li_Lj_coeffs, d0 = 1,
                                  d1 = 1, d2 = 1, d3 = utils.shape(u_e_ij)[1])
    u_e_ij = af.reorder(u_e_ij, 2, 3, 0, 1)

    f_ij_Li_Lj_coeffs = af.broadcast(utils.multiply, u_e_ij,
                                     Li_xi_Lj_eta_coeffs)
    interpolated_f    = af.reorder(af.sum(f_ij_Li_Lj_coeffs, 2),
                                   0, 1, 3, 2)

    return interpolated_f


def volume_integral(u, gv):
    '''
    Vectorize, p, q, moddims.
    '''
    dLp_xi_ij_Lq_eta_ij = gv.dLp_Lq
    dLq_eta_ij_Lp_xi_ij = gv.dLq_Lp

    if (params.volume_integrand_scheme_2d == 'Lobatto' and params.N_LGL == params.N_quad):
        w_i = af.flat(af.transpose(af.tile(gv.lobatto_weights_quadrature, 1, params.N_LGL)))
        w_j = af.tile(gv.lobatto_weights_quadrature, params.N_LGL)
        wi_wj_dLp_xi = af.broadcast(utils.multiply, w_i * w_j, gv.dLp_Lq)
        volume_integrand_ij_1_sp = af.broadcast(utils.multiply,\
                                               wi_wj_dLp_xi, F_xi(u, gv) * gv.sqrt_g)
        wi_wj_dLq_eta = af.broadcast(utils.multiply, w_i * w_j, gv.dLq_Lp)
        volume_integrand_ij_2_sp = af.broadcast(utils.multiply,\
                                               wi_wj_dLq_eta, F_eta(u, gv) * gv.sqrt_g)

        volume_integral = af.reorder(af.sum(volume_integrand_ij_1_sp + volume_integrand_ij_2_sp, 0), 2, 1, 0)

    else: # NEEDS TO BE CHANGED
        volume_integrand_ij_1 = af.broadcast(utils.multiply,\
                                        dLp_xi_ij_Lq_eta_ij,\
                                        F_xi(u, gv))

        volume_integrand_ij_2 = af.broadcast(utils.multiply,\
                                             dLq_eta_ij_Lp_xi_ij,\
                                             F_eta(u, gv))

        volume_integrand_ij = af.moddims((volume_integrand_ij_1 + volume_integrand_ij_2)\
                                        * np.mean(gv.sqrt_det_g), params.N_LGL ** 2,\
                                         (params.N_LGL ** 2) * 100)

        lagrange_interpolation = af.moddims(lag_interpolation_2d(volume_integrand_ij, gv.Li_Lj_coeffs),
                                            params.N_LGL, params.N_LGL, params.N_LGL ** 2  * 100)

        volume_integrand_total = utils.integrate_2d_multivar_poly(lagrange_interpolation[:, :, :],\
                                                    params.N_quad,'gauss', gv)
        volume_integral        = af.transpose(af.moddims(volume_integrand_total, 100, params.N_LGL ** 2))

    return volume_integral

def lax_friedrichs_flux(u, gv):
    '''
    '''
    u = af.reorder(af.moddims(u, params.N_LGL ** 2, 10, 10), 2, 1, 0)

    diff_u_boundary = af.np_to_af_array(np.zeros([10, 10, params.N_LGL ** 2]))

    u_xi_minus1_boundary_right   = u[:, :, :params.N_LGL]
    u_xi_minus1_boundary_left    = af.shift(u[:, :, -params.N_LGL:], d0=0, d1 = 1)
    u[:, :, :params.N_LGL]       = (u_xi_minus1_boundary_right + u_xi_minus1_boundary_left) / 2

    diff_u_boundary[:, :, :params.N_LGL] = (u_xi_minus1_boundary_right - u_xi_minus1_boundary_left)

    u_xi_1_boundary_left  = u[:, :, -params.N_LGL:]
    u_xi_1_boundary_right = af.shift(u[:, :, :params.N_LGL], d0=0, d1=-1)
    u[:, :, :params.N_LGL]     = (u_xi_minus1_boundary_left + u_xi_minus1_boundary_right) / 2

    diff_u_boundary[:, :, -params.N_LGL:] = (u_xi_minus1_boundary_right - u_xi_minus1_boundary_left)


    u_eta_minus1_boundary_down = af.shift(u[:, :, params.N_LGL - 1:params.N_LGL ** 2:params.N_LGL], d0=-1)
    u_eta_minus1_boundary_up   = u[:, :, 0:-params.N_LGL + 1:params.N_LGL]
    u[:, :, 0:-params.N_LGL + 1:params.N_LGL] = (u_eta_minus1_boundary_down\
                                               + u_eta_minus1_boundary_up) / 2
    diff_u_boundary[:, :, 0:-params.N_LGL + 1:params.N_LGL] = (u_eta_minus1_boundary_up\
                                                               -u_eta_minus1_boundary_down)

    u_eta_1_boundary_down = u[:, :, params.N_LGL - 1:params.N_LGL ** 2:params.N_LGL]
    u_eta_1_boundary_up   = af.shift(u[:, :, 0:-params.N_LGL + 1:params.N_LGL], d0=1)

    u[:, :, params.N_LGL - 1:params.N_LGL ** 2:params.N_LGL] = (u_eta_1_boundary_up\
                                                              +u_eta_1_boundary_down) / 2

    diff_u_boundary[:, :, params.N_LGL - 1:params.N_LGL ** 2:params.N_LGL] = (u_eta_1_boundary_up\
                                                                             -u_eta_1_boundary_down)

    u = af.moddims(af.reorder(u, 2, 1, 0), params.N_LGL ** 2, 100)
    diff_u_boundary = af.moddims(af.reorder(diff_u_boundary, 2, 1, 0), params.N_LGL ** 2, 100)
    F_xi_e_ij  = F_xi(u, gv)  - params.c_x * diff_u_boundary
    F_eta_e_ij = F_eta(u, gv) - params.c_y * diff_u_boundary

    return F_xi_e_ij, F_eta_e_ij


def surface_term_vectorized(u, advec_var):
    '''
    '''
    lagrange_coeffs = advec_var.lagrange_coeffs
    N_LGL = params.N_LGL

    eta_LGL = advec_var.xi_LGL


    f_xi_surface_term  = lax_friedrichs_flux(u, advec_var)[0]
    f_eta_surface_term = lax_friedrichs_flux(u, advec_var)[1]

    Lp_xi   = af.moddims(af.reorder(af.tile(utils.polyval_1d(lagrange_coeffs,
                            advec_var.xi_LGL), 1, 1, params.N_LGL), 1, 2, 0), params.N_LGL, 1, params.N_LGL ** 2)

    Lq_eta  = af.tile(af.reorder(utils.polyval_1d(lagrange_coeffs,\
                         eta_LGL), 1, 2, 0), 1, 1, params.N_LGL)

    Lp_xi_1      = af.moddims(af.reorder(af.tile(utils.polyval_1d(lagrange_coeffs, advec_var.xi_LGL[-1]),\
                           1, 1, params.N_LGL), 2, 1, 0), 1, 1, params.N_LGL ** 2)
    Lp_xi_minus1 = af.moddims(af.reorder(af.tile(utils.polyval_1d(lagrange_coeffs, advec_var.xi_LGL[0]),\
                           1, 1, params.N_LGL), 2, 1, 0), 1, 1, params.N_LGL ** 2)

    Lq_eta_1      = af.moddims(af.tile(af.reorder(utils.polyval_1d(lagrange_coeffs,\
                            eta_LGL[-1]), 0, 2, 1), 1, 1, params.N_LGL), 1, 1, params.N_LGL ** 2)
    Lq_eta_minus1 = af.moddims(af.tile(af.reorder(utils.polyval_1d(lagrange_coeffs,\
                             eta_LGL[0]), 0, 2, 1), 1, 1, params.N_LGL), 1, 1, params.N_LGL ** 2)

    # xi = 1 boundary
    Lq_eta_1_boundary   = af.broadcast(utils.multiply, Lq_eta, Lp_xi_1)
    Lq_eta_F_1_boundary = af.broadcast(utils.multiply, Lq_eta_1_boundary,\
                             f_xi_surface_term[-params.N_LGL:, :] * advec_var.sqrt_g[-params.N_LGL:, :])
    Lq_eta_F_1_boundary = af.reorder(Lq_eta_F_1_boundary, 0, 3, 2, 1)


    lag_interpolation_1 = af.sum(af.broadcast(utils.multiply, lagrange_coeffs, Lq_eta_F_1_boundary), 0)
    lag_interpolation_1 = af.reorder(lag_interpolation_1, 2, 1, 3, 0)
    lag_interpolation_1 = af.transpose(af.moddims(af.transpose(lag_interpolation_1),\
                                       params.N_LGL, params.N_LGL ** 2 * 100))

    surface_term_pq_xi_1 = lagrange.integrate(lag_interpolation_1, advec_var) * np.mean(advec_var.sqrt_det_g)
    surface_term_pq_xi_1 = af.moddims(surface_term_pq_xi_1, params.N_LGL ** 2, 100)

    # xi = -1 boundary
    Lq_eta_minus1_boundary   = af.broadcast(utils.multiply, Lq_eta, Lp_xi_minus1)
    Lq_eta_F_minus1_boundary = af.broadcast(utils.multiply, Lq_eta_minus1_boundary,\
                               f_xi_surface_term[:params.N_LGL, :] * advec_var.sqrt_g[:params.N_LGL, :])
    Lq_eta_F_minus1_boundary = af.reorder(Lq_eta_F_minus1_boundary, 0, 3, 2, 1)

    lag_interpolation_2 = af.sum(af.broadcast(utils.multiply, lagrange_coeffs, Lq_eta_F_minus1_boundary), 0)
    lag_interpolation_2 = af.reorder(lag_interpolation_2, 2, 1, 3, 0)
    lag_interpolation_2 = af.transpose(af.moddims(af.transpose(lag_interpolation_2),\
                                       params.N_LGL, params.N_LGL ** 2 * 100))

    surface_term_pq_xi_minus1 = lagrange.integrate(lag_interpolation_2, advec_var)

    surface_term_pq_xi_minus1 = af.moddims(surface_term_pq_xi_minus1, params.N_LGL ** 2, 100)

    # eta = -1 boundary
    Lp_xi_minus1_boundary   = af.broadcast(utils.multiply, Lp_xi, Lq_eta_minus1)
    Lp_xi_F_minus1_boundary = af.broadcast(utils.multiply, Lp_xi_minus1_boundary,\
                              f_eta_surface_term[0:-params.N_LGL + 1:params.N_LGL]\
                              * advec_var.sqrt_g[0:-params.N_LGL + 1:params.N_LGL])
    Lp_xi_F_minus1_boundary = af.reorder(Lp_xi_F_minus1_boundary, 0, 3, 2, 1)

    lag_interpolation_3 = af.sum(af.broadcast(utils.multiply, lagrange_coeffs, Lp_xi_F_minus1_boundary), 0)
    lag_interpolation_3 = af.reorder(lag_interpolation_3, 2, 1, 3, 0)
    lag_interpolation_3 = af.transpose(af.moddims(af.transpose(lag_interpolation_3),\
                                       params.N_LGL, params.N_LGL ** 2 * 100))

    surface_term_pq_eta_minus1 = lagrange.integrate(lag_interpolation_3, advec_var)

    surface_term_pq_eta_minus1 = af.moddims(surface_term_pq_eta_minus1, params.N_LGL ** 2, 100)


    # eta = 1 boundary
    Lp_xi_1_boundary   = af.broadcast(utils.multiply, Lp_xi, Lq_eta_1)
    Lp_xi_F_1_boundary = af.broadcast(utils.multiply, Lp_xi_1_boundary,\
                         f_eta_surface_term[params.N_LGL - 1:params.N_LGL ** 2:params.N_LGL]\
                         * advec_var.sqrt_g[params.N_LGL - 1:params.N_LGL ** 2:params.N_LGL])
    Lp_xi_F_1_boundary = af.reorder(Lp_xi_F_1_boundary, 0, 3, 2, 1)

    lag_interpolation_4 = af.sum(af.broadcast(utils.multiply, lagrange_coeffs, Lp_xi_F_1_boundary), 0)
    lag_interpolation_4 = af.reorder(lag_interpolation_4, 2, 1, 3, 0)
    lag_interpolation_4 = af.transpose(af.moddims(af.transpose(lag_interpolation_4),\
                                       params.N_LGL, params.N_LGL ** 2 * 100))

    surface_term_pq_eta_1 = lagrange.integrate(lag_interpolation_4, advec_var)

    surface_term_pq_eta_1 = af.moddims(surface_term_pq_eta_1, params.N_LGL ** 2, 100)

    surface_term_e_pq = surface_term_pq_xi_1\
                      - surface_term_pq_xi_minus1\
                      + surface_term_pq_eta_1\
                      - surface_term_pq_eta_minus1

    return surface_term_e_pq



def b_vector(u, advec_var):
    '''
    '''
    surface_term_u_pq    = surface_term_vectorized(u, advec_var)
    volume_integral_pq   = volume_integral(u, advec_var)
    b_vector_array       = volume_integral_pq - surface_term_u_pq

    return b_vector_array


def RK4_timestepping(A_inverse, u, delta_t, gv):
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

    k1 = af.matmul(A_inverse, b_vector(u, gv))
    k2 = af.matmul(A_inverse, b_vector(u + k1 * delta_t / 2, gv))
    k3 = af.matmul(A_inverse, b_vector(u + k2 * delta_t / 2, gv))
    k4 = af.matmul(A_inverse, b_vector(u + k3 * delta_t    , gv))

    delta_u = delta_t * (k1 + 2 * k2 + 2 * k3 + k4) / 6

    return delta_u

def time_evolution(gv):
    '''
    '''
    # Creating a folder to store hdf5 files. If it doesn't exist.
    results_directory = 'results/xi_eta_2d_hdf5_%02d' %(int(params.N_LGL))
    if not os.path.exists(results_directory):
        os.makedirs(results_directory)

    A_inverse = af.np_to_af_array(np.linalg.inv(np.array(A_matrix_xi_eta(params.N_LGL, gv))))
    xi_LGL = lagrange.LGL_points(params.N_LGL)
    xi_i   = af.flat(af.transpose(af.tile(xi_LGL, 1, params.N_LGL)))
    eta_j  = af.tile(xi_LGL, params.N_LGL)

    u_init_2d = gv.u_e_ij
    delta_t   = gv.delta_t_2d
    time      = gv.time
    u         = u_init_2d
    time      = gv.time_2d
    
    for i in trange(0, time.shape[0]):
        L1_norm = af.mean(af.abs(u_init_2d - u))

        if (L1_norm >= 100):
            print(L1_norm)
            break
        if (i % 10) == 0:
            h5file = h5py.File('results/xi_eta_2d_hdf5_%02d/dump_timestep_%06d' %(int(params.N_LGL), int(i)) + '.hdf5', 'w')
            dset   = h5file.create_dataset('u_i', data = u, dtype = 'd')

            dset[:, :] = u[:, :]

        u += RK4_timestepping(A_inverse, u, delta_t, gv)

    return L1_norm
