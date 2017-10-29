#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import arrayfire as af

from dg_maxwell import params
from dg_maxwell import isoparam
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


def A_matrix():
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

    N_LGL   = params.N_LGL
    xi_LGL  = lagrange.LGL_points(N_LGL)
    eta_LGL = lagrange.LGL_points(N_LGL)

    _, Lp = lagrange.lagrange_polynomials(xi_LGL)
    Lp = af.np_to_af_array(Lp)
    Li = Lp.copy()

    _, Lq = lagrange.lagrange_polynomials(eta_LGL)
    Lq = af.np_to_af_array(Lq)
    Lj = Lq.copy()

    Li = af.reorder(Li, d0 = 0, d1 = 2, d2 = 1)
    Li = af.transpose(af.tile(Li, d0 = 1, d1 = N_LGL))
    Li = af.moddims(Li, d0 = N_LGL * N_LGL, d1 = 1, d2 = N_LGL)
    Li = af.transpose(af.tile(Li, d0 = 1, d1 = N_LGL * N_LGL))
    Li = af.transpose(af.moddims(af.transpose(Li),
                                 d0 = N_LGL * N_LGL * N_LGL * N_LGL,
                                 d1 = 1, d2 = N_LGL))
    Li = af.reorder(Li, d0 = 2, d1 = 1, d2 = 0)

    Lp = af.reorder(Lp, d0 = 0, d1 = 2, d2 = 1)
    Lp = af.transpose(af.tile(Lp, d0 = 1, d1 = N_LGL))
    Lp = af.moddims(Lp, d0 = N_LGL * N_LGL, d1 = 1, d2 = N_LGL)
    Lp = af.tile(Lp, d0 = 1, d1 = N_LGL * N_LGL)
    Lp = af.moddims(af.transpose(Lp),
                    d0 = N_LGL * N_LGL * N_LGL * N_LGL,
                    d1 = 1, d2 = N_LGL)
    Lp = af.reorder(af.transpose(Lp), d0 = 2, d1 = 1, d2 = 0)

    Lp_Li = af.transpose(af.convolve1(Li, Lp, conv_mode = af.CONV_MODE.EXPAND))

    Lj = af.reorder(Lj, d0 = 0, d1 = 2, d2 = 1)
    Lj = af.tile(Lj, d0 = 1, d1 = N_LGL)
    Lj = af.moddims(Lj, d0 = N_LGL * N_LGL, d1 = 1, d2 = N_LGL)
    Lj = af.transpose(af.tile(Lj, d0 = 1, d1 = N_LGL * N_LGL))
    Lj = af.moddims(af.transpose(Lj),
                    d0 = N_LGL * N_LGL * N_LGL * N_LGL,
                    d1 = 1, d2 = N_LGL)
    Lj = af.reorder(Lj, d0 = 2, d1 = 0, d2 = 1)

    Lq = af.reorder(Lq, d0 = 0, d1 = 2, d2 = 1)
    Lq = af.tile(Lq, d0 = 1, d1 = N_LGL)
    Lq = af.moddims(Lq, d0 = N_LGL * N_LGL, d1 = 1, d2 = N_LGL)
    Lq = af.tile(Lq, d0 = 1, d1 = N_LGL * N_LGL)
    Lq = af.moddims(af.transpose(Lq),
                    d0 = N_LGL * N_LGL * N_LGL * N_LGL,
                    d1 = 1, d2 = N_LGL)
    Lq = af.reorder(af.transpose(Lq), d0 = 2, d1 = 1, d2 = 0)

    Lq_Lj = af.transpose(af.convolve1(Lj, Lq, conv_mode = af.CONV_MODE.EXPAND))

    A = af.moddims(utils.integrate_2d(Lp_Li, Lq_Lj,
                                      order = 9,
                                      scheme = 'gauss'),
                   d0 = N_LGL * N_LGL, d1 = N_LGL * N_LGL)

    return A


def u_init():
    '''
    The boundary flux
    '''

    element_LGL = isoparam.u_pq_isoparam()
    element_LGL = af.moddims(element_LGL, params.N_LGL ** 2, 2, 3, 3)
    element_LGL = af.reorder(element_LGL, 2, 3, 0, 1)
    element_LGL = af.flip(element_LGL, dim=1)

    element_LGL_initialize = af.reorder(element_LGL, 2, 3, 1, 0)

    #u_init = np.e ** (-(element_LGL_initialize[0], 

    return element_LGL


def flux_x(u):
    '''
    The flux in x direction
    '''

    flux = params.c_x * u

    return flux


def boundary_flux(u):
    '''
    The flux at the element boundaries, calculated using the Lax-Friedrichs method.
    '''
    u_right_boundary_i      = u[:, :, -params.N_LGL:, :]
    u_left_boundary_i_plus1 = af.shift(u[:, :, :params.N_LGL, :], -1)
    flux_vert_boundary      = (flux_x(u_left_boundary_i_plus1) \
                                + flux_x(u_right_boundary_i)) / 2 \
                                + params.c_lax * (u_left_boundary_i_plus1\
                                - u_right_boundary_i)

    return flux_vert_boundary



def surface_term():
    '''
    The surface term obtained for 2D advection.
    '''
    
    return
