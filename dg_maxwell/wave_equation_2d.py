#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import arrayfire as af
af.set_backend('cpu')

from dg_maxwell import params
from dg_maxwell import msh_parser
from dg_maxwell import lagrange
from dg_maxwell import utils

from tqdm import trange

import os
import sys
import csv
sys.path.insert(0, os.path.abspath('../'))

af.set_backend(params.backend)

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


def A_matrix(N_LGL):
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
    print(af.info())
    N_LGL = params.N_LGL
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
                                                     
    A = utils.integrate_2d_multivar_poly(Lp_Li_Lq_Lj_tp,
                                         N_quad = params.N_LGL, scheme = 'gauss')
    
    A = af.moddims(A, d0 = N_LGL * N_LGL, d1 = N_LGL * N_LGL)

    return A


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


def sqrtgDet(x_nodes, y_nodes, xi, eta):
    '''
    '''
    gCov = g_dd(x_nodes, y_nodes, xi, eta)
    
    a = gCov[0][0]
    b = gCov[0][1]
    c = gCov[1][0]
    d = gCov[1][1]
    
    return (a*d - b*c)**0.5

def F_xi(u):
    '''
    '''
    nodes, elements = msh_parser.read_order_2_msh('square_1.msh')

    xi_LGL = lagrange.LGL_points(params.N_LGL)
    xi_i   = af.flat(af.transpose(af.tile(xi_LGL, params.N_LGL)))
    eta_j  = af.tile(xi_LGL, params.N_LGL)

    dxi_by_dx = dxi_dx(nodes[elements[0]][:, 0], nodes[elements[0]][:, 1], xi_i, eta_j)
    dxi_by_dy = dxi_dy(nodes[elements[0]][:, 0], nodes[elements[0]][:, 1], xi_i, eta_j)
    F_xi_u = F_x(u) * dxi_by_dx + F_y(u) * dxi_by_dy

    return F_xi_u


def F_eta(u):
    '''
    '''
    nodes, elements = msh_parser.read_order_2_msh('square_1.msh')

    xi_LGL = lagrange.LGL_points(params.N_LGL)
    xi_i   = af.flat(af.transpose(af.tile(xi_LGL, params.N_LGL)))
    eta_j  = af.tile(xi_LGL, params.N_LGL)

    deta_by_dx = deta_dx(nodes[elements[0]][:, 0], nodes[elements[0]][:, 1], xi_i, eta_j)
    deta_by_dy = deta_dy(nodes[elements[0]][:, 0], nodes[elements[0]][:, 1], xi_i, eta_j)
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

def lag_interpolation_2d(f_ij, N_LGL):
    '''
    '''
    Li_xi_Lj_eta_coeffs = Li_Lj_coeffs(N_LGL)
    f_ij = af.reorder(f_ij, 2, 1, 0)

    f_ij_Li_Lj_coeffs = af.broadcast(utils.multiply, f_ij, Li_xi_Lj_eta_coeffs)
    interpolated_f    = af.sum(f_ij_Li_Lj_coeffs, 2)

    return interpolated_f


def volume_integral(u, N_quadrature, int_scheme):
    '''
    '''
    nodes, elements = msh_parser.read_order_2_msh('square_1.msh')

    xi_i   = af.flat(af.transpose(af.tile(params.xi_LGL, 1, params.N_LGL)))
    eta_j  = af.tile(params.xi_LGL, params.N_LGL)

    u_ij    = u

    dLp_dxi = af.moddims(af.tile(af.reorder(params.dl_dxi_coeffs, 2, 0, 1), params.N_LGL), params.N_LGL ** 2, params.N_LGL - 1)
    Lq_eta  = af.tile(params.lagrange_coeffs, params.N_LGL)
    g_ab    = g_uu(nodes[elements[0]][:, 0], nodes[elements[0]][:, 1], np.array(xi_i), np.array(eta_j))

    volume_integral = af.np_to_af_array(np.zeros([params.N_LGL ** 2]))
#    dlp_dxi_ij_v = af.reorder(utils.polyval_1d(dLp_dxi, xi_i), 2, 3, 1, 0)
#    Lq_eta_ij_v  = af.reorder(utils.polyval_1d(Lq_eta, eta_j), 2, 3, 1, 0)
#    F_xi_v       = af.reorder(F_xi(u_ij), 2, 3, 0, 1)
#    vol_int_pq_v = af.np_to_af_array(np.zeros([params.N_LGL, params.N_LGL, params.N_LGL ** 2]))
#    a = (af.reorder(utils.polyval_1d(dLp_dxi[0], xi_i), 2, 3, 1, 0))
#    b = (af.reorder(utils.polyval_1d(Lq_eta[0], eta_j), 2, 3, 1, 0))
#    c = (F_xi_v)
#    d = (Li_Lj_coeffs(params.N_LGL))
#    #print(af.broadcast(utils.multiply, a * b * c, d).shape)
#    ii = 0
#    print(af.sum(af.broadcast(utils.multiply, af.reorder(utils.polyval_1d(dLp_dxi[ii], xi_i), 2, 3, 1, 0) 
#        * af.reorder(utils.polyval_1d(Lq_eta[ii], eta_j), 2, 3, 1, 0)
#        * F_xi_v, Li_Lj_coeffs(params.N_LGL)), 2).shape)



    for p in range(params.N_LGL):
        for q in range(params.N_LGL):
            index = p * params.N_LGL + q
            dLp_dxi_ij = af.transpose(utils.polyval_1d(dLp_dxi[index], xi_i))
            Lq_eta_ij  = af.transpose(utils.polyval_1d(Lq_eta[index], eta_j))

            volume_integral_pq   = F_xi(u_ij) * dLp_dxi_ij * Lq_eta_ij
            vol_int_interpolated = lag_interpolation_2d(volume_integral_pq, params.N_LGL)
            volume_integral[index] = utils.integrate_2d_multivar_poly(vol_int_interpolated, N_quad = N_quadrature, scheme = int_scheme)


    return volume_integral

def analytical_vol_int():
    '''
    '''
    csv_handler = csv.reader(open('volume_integral_pq_2d_analytical.csv',
                                  newline='\n'), delimiter=',')
    
    content = list()
    
    for n, line in enumerate(csv_handler):
        content.append(list())
        for item in line:
            try:
                content[-1].append(float(item))
            except ValueError:
                if content[-1] == []:
                    content.pop()
                    print('popping string')
                break
    
    volume_integral_pq_analytical = af.np_to_af_array(np.array(content))
    return volume_integral_pq_analytical

def lax_friedrichs_flux(u):
    '''
    '''

    diff_u_boundary = af.np_to_af_array(np.zeros([params.N_LGL ** 2]))

    u_xi_1_boundary_right = u[:params.N_LGL]
    u_xi_1_boundary_left  = u[-params.N_LGL:]
    u[-params.N_LGL:]     = (u_xi_1_boundary_right + u_xi_1_boundary_left) / 2

    diff_u_boundary[-params.N_LGL:]  = (u_xi_1_boundary_right - u_xi_1_boundary_left)

    u_xi_minus1_boundary_left  = u[-params.N_LGL:]
    u_xi_minus1_boundary_right = u[:params.N_LGL]
    u[:params.N_LGL]           = (u_xi_minus1_boundary_right + u_xi_minus1_boundary_left) / 2

    diff_u_boundary[:params.N_LGL]  = (u_xi_minus1_boundary_right - u_xi_minus1_boundary_left)

    u_eta_1_boundary_up   = u[params.N_LGL - 1:params.N_LGL ** 2:params.N_LGL]
    u_eta_1_boundary_down = u[0:-params.N_LGL + 1:params.N_LGL]

    u[params.N_LGL - 1:params.N_LGL ** 2:params.N_LGL] = (u_eta_1_boundary_up + u_eta_1_boundary_down) / 2

    diff_u_boundary[params.N_LGL - 1:params.N_LGL ** 2:params.N_LGL] = (u_eta_1_boundary_up - u_eta_1_boundary_down)

    
    u_eta_minus1_boundary_down   = u[params.N_LGL - 1:params.N_LGL ** 2:params.N_LGL]
    u_eta_minus1_boundary_up     = u[0:-params.N_LGL + 1:params.N_LGL]


    u[0:-params.N_LGL + 1:params.N_LGL] = (u_eta_minus1_boundary_down + u_eta_minus1_boundary_up) / 2
    diff_u_boundary[0:-params.N_LGL + 1:params.N_LGL] = (u_eta_minus1_boundary_down - u_eta_minus1_boundary_down)

    F_xi_element  = F_xi(u)  - params.c_lax * diff_u_boundary
    F_eta_element = F_eta(u) - params.c_lax * diff_u_boundary


    return F_xi_element, F_eta_element

def surface_term(u):
    '''
    '''
    nodes, elements = msh_parser.read_order_2_msh('square_1.msh')

    xi_LGL  = lagrange.LGL_points(params.N_LGL)
    eta_LGL = lagrange.LGL_points(params.N_LGL)
    xi_i    = af.flat(af.transpose(af.tile(xi_LGL, 1, params.N_LGL)))
    eta_j   = af.tile(xi_LGL, params.N_LGL)

    lagrange_coeffs = af.np_to_af_array(lagrange.lagrange_polynomials(xi_LGL)[1])


    Lp_coeffs = af.moddims(af.tile(af.reorder(af.transpose(lagrange_coeffs), 2, 1, 0),\
                              params.N_LGL), params.N_LGL ** 2, params.N_LGL)
    Lp_1      = utils.polyval_1d(Lp_coeffs, xi_LGL[-1])
    Lp_minus1 = utils.polyval_1d(Lp_coeffs, xi_LGL[0])


#    Lq_coeffs = af.tile(lagrange_coeffs, params.N_LGL)
#    Lq_1      = utils.polyval_1d(Lq_coeffs, xi_LGL[-1])
#    Lq_minus1 = utils.polyval_1d(Lq_coeffs, xi_LGL[0])

    F_xi_surface_term = F_xi(u)

    g_ab = g_uu(nodes[elements[0]][:, 0], nodes[elements[0]][:, 1], np.array(xi_i), np.array(eta_j))
    g_00 = (af.np_to_af_array(g_ab[0][0]))
    g_01 = (af.np_to_af_array(g_ab[0][1]))
    g_10 = (af.np_to_af_array(g_ab[1][0]))
    g_11 = (af.np_to_af_array(g_ab[1][1]))

    F_xi_element  = lax_friedrichs_flux(u)[0]
    nodes    = params.nodes
    elements = params.elements
    F_eta_element = lax_friedrichs_flux(u)[1]
    surface_term_pq  = af.np_to_af_array(np.zeros([params.N_LGL ** 2]))

    for p in range(params.N_LGL):
        Lp_1 = utils.polyval_1d(lagrange_coeffs[p], xi_LGL[-1])
        Lp_minus1 = utils.polyval_1d(lagrange_coeffs[p], xi_LGL[0])
        for q in range(params.N_LGL):

            Lq_1      = utils.polyval_1d(lagrange_coeffs[q], eta_LGL[-1])
            Lq_minus1 = utils.polyval_1d(lagrange_coeffs[q], eta_LGL[0])
            Lq_eta    = af.transpose(utils.polyval_1d(lagrange_coeffs[q], eta_LGL))
            Lp_xi = af.transpose(utils.polyval_1d(lagrange_coeffs[p], xi_LGL))

            # The First integral
            Lq_eta_F_xi_LGL      = Lq_eta * F_xi_element[-params.N_LGL:] * g_00[-params.N_LGL:]
            lag_interpolation_1  = af.sum(af.broadcast(utils.multiply, Lq_eta_F_xi_LGL, lagrange_coeffs), 0)
            surface_term_pq_xi_1 = af.sum(Lp_1) * lagrange.integrate((lag_interpolation_1))


            # The second integral
            Lp_xi_F_xi_LGL = Lp_xi * F_xi_element[params.N_LGL - 1:params.N_LGL ** 2:params.N_LGL]\
                                   * g_01[params.N_LGL - 1:params.N_LGL ** 2:params.N_LGL]

            lag_interpolation_2 = af.sum(af.broadcast(utils.multiply, Lp_xi_F_xi_LGL, lagrange_coeffs), 0)
            surface_term_pq_eta_1    = af.sum(Lq_1) * lagrange.integrate(lag_interpolation_2)

            # The third integral
            Lq_eta_F_xi_LGL     = Lq_eta * F_xi_element[:params.N_LGL] * g_00[:params.N_LGL]
            lag_interpolation_3 = af.sum(af.broadcast(utils.multiply, Lq_eta_F_xi_LGL, lagrange_coeffs), 0)
            surface_term_pq_xi_minus1 = af.sum(Lp_minus1) * lagrange.integrate((lag_interpolation_3))

            # The fourth integral
            Lp_xi_F_xi_LGL = Lp_xi * F_xi_element[0:-params.N_LGL + 1:params.N_LGL]\
                                   * g_01[0:-params.N_LGL + 1:params.N_LGL]

            lag_interpolation_4 = af.sum(af.broadcast(utils.multiply, Lp_xi_F_xi_LGL, lagrange_coeffs), 0)
            surface_term_pq_eta_1    = af.sum(Lq_1) * lagrange.integrate(lag_interpolation_2)

            surface_term_pq[p * params.N_LGL + q]  =  surface_term_pq_eta_1\
                                                    + surface_term_pq_xi_minus1\
                                                    + surface_term_pq_xi_1

    return surface_term_pq


def b_vector(u):
    '''
    '''
    surface_term_u_pq    = surface_term(u)
    volume_integral_pq   = volume_integral(u, params.N_LGL + 1, 'gauss')
    b_vector_array       = surface_term_u_pq - volume_integral_pq

    return b_vector_array


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

def time_evolution():
    '''
    '''
    A_inverse = af.inverse(A_matrix(params.N_LGL))
    xi_LGL = lagrange.LGL_points(params.N_LGL)
    xi_i   = af.flat(af.transpose(af.tile(xi_LGL, 1, params.N_LGL)))
    eta_j  = af.tile(xi_LGL, params.N_LGL)

    u_init_2d = np.e ** (- (xi_i ** 2) / (0.6 ** 2))
    delta_t   = params.delta_t
    time      = params.time
    u         = u_init_2d
    print(A_inverse)
    print(b_vector(u))
    
    for t_n in trange(0, 1):
        u += RK4_timestepping(A_inverse, u, delta_t)
        print(u)
