#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import arrayfire as af
import numpy as np
import os
from matplotlib import pyplot as pl

from dg_maxwell import utils
from dg_maxwell import params
from dg_maxwell import lagrange
from dg_maxwell import msh_parser

af.set_backend(params.backend)
af.set_device(params.device)

def isoparam_1D(x_nodes, xi):
    '''
    Maps points in :math:`\\xi` space to :math:`x` space using the formula
    :math:`x = \\frac{1 - \\xi}{2} x_0 + \\frac{1 + \\xi}{2} x_1`

    Parameters
    ----------

    x_nodes : arrayfire.Array [2 1 1 1]
              Element nodes.

    xi      : arrayfire.Array [N 1 1 1]
              Value of :math:`\\xi` coordinate for which the corresponding
              :math:`x` coordinate is to be found.

    Returns
    -------
    x : arrayfire.Array
        :math:`x` value in the element corresponding to :math:`\\xi`.
    '''
    N_0 = (1 - xi) / 2
    N_1 = (1 + xi) / 2

    N0_x0 = af.broadcast(utils.multiply, N_0, x_nodes[0])
    N1_x1 = af.broadcast(utils.multiply, N_1, x_nodes[1])

    x = N0_x0 + N1_x1

    return x


def b_isoparam_x_2D(x_nodes, xi, eta):
    '''

    Finds the :math:`x` coordinate using isoparametric mapping

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

    x_nodes = af.reorder(x_nodes, 2, 1, 0)

    x = af.np_to_af_array(np.zeros([64, 1, x_nodes.shape[2]]))

    for ii in range(x_nodes.shape[2]):
        x[:, 0, ii] =   N_0 * af.sum(x_nodes[0, 0, ii]) \
                      + N_1 * af.sum(x_nodes[0, 1, ii]) \
                      + N_2 * af.sum(x_nodes[0, 2, ii]) \
                      + N_3 * af.sum(x_nodes[0, 3, ii]) \
                      + N_4 * af.sum(x_nodes[0, 4, ii]) \
                      + N_5 * af.sum(x_nodes[0, 5, ii]) \
                      + N_6 * af.sum(x_nodes[0, 6, ii]) \
                      + N_7 * af.sum(x_nodes[0, 7, ii])

    return x


def isoparam_x_2D(x_nodes, xi, eta):
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

    x = N_0 * (x_nodes[0]) \
      + N_1 * (x_nodes[1]) \
      + N_2 * (x_nodes[2]) \
      + N_3 * (x_nodes[3]) \
      + N_4 * (x_nodes[4]) \
      + N_5 * (x_nodes[5]) \
      + N_6 * (x_nodes[6]) \
      + N_7 * (x_nodes[7])


    return x


def isoparam_y_2D(y_nodes, xi, eta):
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
    y = isoparam_x_2D(y_nodes, xi, eta)
    return y


def u_pq_isoparam():
    '''
    '''
    nodes, elements = msh_parser.read_order_2_msh(\
		      os.path.abspath('./dg_maxwell/tests/mesh/Square_N_Elements_9.msh'))

    axes_handler = pl.axes()
    xi_LGL       = np.array(lagrange.LGL_points(params.N_LGL))
    eta_LGL      = np.array(lagrange.LGL_points(params.N_LGL))
    #msh_parser.plot_mesh_grid(nodes, elements, xi_LGL, eta_LGL, axes_handler)
    pl.xlim(-3, 3)
    pl.ylim(-3, 3)

    xi  = af.flat(af.transpose(af.tile(params.xi_LGL, 1, params.N_LGL)))
    eta = af.tile(params.xi_LGL, params.N_LGL)

    u_pq = af.np_to_af_array(np.zeros([params.N_LGL ** 2, 2, 9]))

    for i in range(9):

        element_nodes_number = elements[i][:-1]
        element_nodes_number = np.roll(element_nodes_number, -2)
        element_nodes        = nodes[element_nodes_number,:]
        element_nodes        = af.np_to_af_array(element_nodes)

        x_nodes = element_nodes[:, 0]
        y_nodes = element_nodes[:, 1]

        mapped_coords       = af.np_to_af_array(np.zeros([params.N_LGL ** 2, 2]))
        mapped_coords[:, 0] = isoparam_x_2D(x_nodes, xi, eta)
        mapped_coords[:, 1] = isoparam_x_2D(y_nodes, xi, eta)

        u_pq[:, 0, i] = mapped_coords[:, 0]
        u_pq[:, 1, i] = mapped_coords[:, 1]
        pl.scatter(np.array(mapped_coords[:, 0]), np.array(mapped_coords[:, 1]))

    pl.scatter(np.array(u_pq[:, 0, :]), np.array(u_pq[:, 1, :]))

    return u_pq
