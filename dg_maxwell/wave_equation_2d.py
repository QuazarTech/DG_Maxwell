#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import arrayfire as af

from dg_maxwell import params
from dg_maxwell import lagrange
from dg_maxwell import utils

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
                                                     
    A = []
    for index in np.arange(Lp_Li_Lq_Lj_tp.shape[2]):
        print(index)
        A.append(utils.integrate_2d_multivar_poly(Lp_Li_Lq_Lj_tp[:, :, index],
                                                N_quad = 9, scheme = 'gauss'))
    
    A = af.np_to_af_array(np.array(A))
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


def c_xi(x_nodes, y_nodes, xi, eta):
    '''
    '''
    c_xi = (params.c_x * dy_deta(y_nodes, xi, eta)  \
          - params.c_y * dx_deta(x_nodes, xi, eta)) \
            (dx_dxi(x_nodes, xi, eta) * dy_deta(y_nodes, xi, eta) \
          -  dx_deta(x_nodes, xi, eta) * dy_dxi(y_nodes, xi, eta))
        
    return c_xi


def c_eta(x_nodes, y_nodes, xi, eta):
    '''
    '''
    c_eta =  (params.c_x * dy_dxi(y_nodes, xi, eta)  \
           -  params.c_y * dx_dxi(x_nodes, xi, eta)) \
           / (dx_deta(x_nodes, xi, eta) * dy_dxi(y_nodes, xi, eta) \
           -  dx_dxi(x_nodes, xi, eta) * dy_deta(y_nodes, xi, eta))
        
    return c_eta


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
    
    ans = np.array([[d, -b],
                    [-c, a]])
    
    return np.array(ans)/det


def sqrtgDet(x_nodes, y_nodes, xi, eta):
    '''
    '''
    gCov = g_dd(x_nodes, y_nodes, xi, eta)
    
    a = gCov[0][0]
    b = gCov[0][1]
    c = gCov[1][0]
    d = gCov[1][1]
    
    return (a*d - b*c)**0.5

def sqrtgDet(x_nodes, y_nodes, xi, eta):
    '''
    '''
    gCov = g_dd(x_nodes, y_nodes, xi, eta)
    
    a = gCov[0][0]
    b = gCov[0][1]
    c = gCov[1][0]
    d = gCov[1][1]
    
    return (a*d - b*c)**0.5


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
