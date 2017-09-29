#! /usr/bin/env python3
# -*- coding: utf-8 -*-

def isoparam_x(x_nodes, xi, eta):
    '''
    Finds the :math:`x` coordinate using isoparametric mapping of a
    :math:`2^{nd}` order element with :math:`8` nodes
    .. math:: (P_0, P_1, P_2, P_3, P_4, P_5, P_6, P_7)

    Here :math:`P_i` corresponds to :math:`(\\xi_i, \\eta_i)` coordinates,
    :math:`i \in \\{0, 1, ..., 7\\}` respectively, where,
    
    .. math:: (\\xi_0, \\eta_0) &\equiv (-1,  1) \\\\
              (\\xi_1, \\eta_1) &\equiv (-1,  0) \\\\
              (\\xi_2, \\eta_2) &\equiv (-1, -1) \\\\
              (\\xi_3, \\eta_3) &\equiv ( 0, -1) \\\\
              (\\xi_4, \\eta_4) &\equiv ( 1, -1) \\\\
              (\\xi_5, \\eta_5) &\equiv ( 1,  0) \\\\
              (\\xi_6, \\eta_6) &\equiv ( 1,  1) \\\\
              (\\xi_7, \\eta_7) &\equiv ( 0,  1)
              
    Parameters
    ----------
    x_nodes : np.ndarray [8]
              :math:`x` nodes.
              
    xi      : float
            :math:`\\xi` coordinate for which :math:`x` has to be found.

    eta     : float
            :math:`\\eta` coordinate for which :math:`x` has to be found.

    Returns
    -------
    x : float
        :math:`x` coordinate corresponding to :math:`(\\xi, \\eta)` coordinate.
    
    '''
    N_0 = (-1.0 / 4.0) * (1 - xi)  * (1 + eta) * (1 + xi - eta)
    N_1 = (1.0 / 2.0)  * (1 - xi)  * (1 - eta**2)
    N_2 = (-1.0 / 4.0) * (1 - xi)  * (1 - eta) * (1 + xi + eta)
    N_3 = (1.0 / 2.0)  * (1 - eta) * (1 - xi**2)
    N_4 = (-1.0 / 4.0) * (1 + xi)  * (1 - eta) * (1 - xi + eta)
    N_5 = (1.0 / 2.0)  * (1 + xi)  * (1 - eta**2)
    N_6 = (-1.0 / 4.0) * (1 + xi)  * (1 + eta) * (1 - xi - eta)
    N_7 = (1.0 / 2.0)  * (1 + eta) * (1 - xi**2)
    
    x = N_0 * x_nodes[0] \
      + N_1 * x_nodes[1] \
      + N_2 * x_nodes[2] \
      + N_3 * x_nodes[3] \
      + N_4 * x_nodes[4] \
      + N_5 * x_nodes[5] \
      + N_6 * x_nodes[6] \
      + N_7 * x_nodes[7]
    
    return x


def isoparam_y(y_nodes, xi, eta):
    '''
    This function allows isoparametric mapping of a :math:`2^{nd}` order
    element with :math:`8` nodes
    
    .. math:: (P_0, P_1, P_2, P_3, P_4, P_5, P_6, P_7)
    
    Here :math:`P_i` corresponds to :math:`(\\xi_i, \\eta_i)` coordinates,
    :math:`i \in \\{0, 1, ..., 7\\}` respectively, where,
    
    .. math:: (\\xi_0, \\eta_0) &\equiv (-1,  1) \\\\
              (\\xi_1, \\eta_1) &\equiv (-1,  0) \\\\
              (\\xi_2, \\eta_2) &\equiv (-1, -1) \\\\
              (\\xi_3, \\eta_3) &\equiv ( 0, -1) \\\\
              (\\xi_4, \\eta_4) &\equiv ( 1, -1) \\\\
              (\\xi_5, \\eta_5) &\equiv ( 1,  0) \\\\
              (\\xi_6, \\eta_6) &\equiv ( 1,  1) \\\\
              (\\xi_7, \\eta_7) &\equiv ( 0,  1)
              
    Parameters
    ----------
    y_nodes : np.ndarray [8]
              :math:`y` nodes.
              
    xi      : float
            :math:`\\xi` coordinate for which :math:`y` has to be found.

    eta     : float
            :math:`\\eta` coordinate for which :math:`y` has to be found.

    Returns
    -------
    float
        :math:`y` coordinate corresponding to :math:`(\\xi, \\eta)` coordinate.
    
    '''
    return isoparam_x(y_nodes, xi, eta)
