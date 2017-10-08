#! /usr/bin/env python3
# -*- coding: utf-8 -*-

from dg_maxwell import isoparam


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
              
    xi      : float
            :math:`\\xi` coordinate for which
            :math:`\\frac{\\partial x}{\\partial \\xi}` has to be found.

    eta     : float
            :math:`\\eta` coordinate for which
            :math:`\\frac{\\partial x}{\\partial \\xi}` has to be found.
            
    Returns
    -------
    dx_dxi : float
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
              
    xi      : float
            :math:`\\xi` coordinate for which
            :math:`\\frac{\\partial x}{\\partial \\eta}` has to be found.

    eta     : float
            :math:`\\eta` coordinate for which
            :math:`\\frac{\\partial x}{\\partial \\eta}` has to be found.
            
    Returns
    -------
    dx_deta : float
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
              
    xi      : float
            :math:`\\xi` coordinate for which
            :math:`\\frac{\\partial y}{\\partial \\xi}` has to be found.

    eta     : float
            :math:`\\eta` coordinate for which
            :math:`\\frac{\\partial y}{\\partial \\xi}` has to be found.
            
    Returns
    -------
    float
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
              
    xi      : float
            :math:`\\xi` coordinate for which
            :math:`\\frac{\\partial y}{\\partial \\eta}` has to be found.

    eta     : float
            :math:`\\eta` coordinate for which
            :math:`\\frac{\\partial y}{\\partial \\eta}` has to be found.
            
    Returns
    -------
    float
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
              
    xi      : float
            :math:`\\xi` coordinate at which
            Jacobian has to be found.

    eta     : float
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
    
    dx_dxi  = dx_dxi (x_nodes, xi, eta)
    dy_deta = dy_deta (y_nodes, xi, eta)
    dx_deta = dx_deta (x_nodes, xi, eta)
    dy_dxi  = dy_dxi (y_nodes, xi, eta)
    
    return (dx_dxi * dy_deta) - (dx_deta * dy_dxi)