#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.lines as lines
import gmshtranslator.gmshtranslator as gmsh

import msh_parser
import isoparam
import utils

def read_order_2_msh(msh_file):
    '''
    Parses the :math:`2^{nd}` order **.msh** files.
    
    Parameters
    ----------
    
    msh_file : str
               **.msh** file to be parsed
    
    Returns
    -------
    
    tuple(np.ndarray, np.ndarray)
             Tuple of Nodes and Elements respectively.
             Nodes is a :math:`N \\times 2` array, where, :math:`N` is the
             total number of Nodes in the mesh. Each node contains it's
             :math:`(x, y)` coordinates.
             Elements is a :math:`N_e \\times 8` array which contains the tags
             of all the nodes which defines each element. A tag of a node is
             the array index of a node.
    '''
    
    msh_handler = gmsh.gmshTranslator(msh_file)
    
    nodes    = []
    elements = []
    
    def is_node(tag, x, y, z, physgroups):
        return True
    
    def save_node (tag, x, y, z):
        nodes.append([x, y])
        return

    
    def is_9_node_quadrangle (eletag, eletype,
                              physgrp, nodes):
        return eletype == msh_handler.quadrangle_9_node

    def save_element (eletag, eletype,
                      physgrp, node_tags):
        
        temp_nodes = node_tags.copy()
        for j, k in zip(np.arange (0,8,2), np.arange(4)):
            node_tags[j]     = temp_nodes[k]
            node_tags[j + 1] = temp_nodes[k + 4]
        
        # The node tag starts from 1, but now they will start from 0
        # because the nodes array indices represent the node tag.
        # Therefore (node_tags - 1) instead of (node_tags)
        elements.append(node_tags - 1)
    
    msh_handler.add_nodes_rule (is_node, save_node)
    msh_handler.parse()
    
    msh_handler.clear_rules()
    msh_handler.add_elements_rule (is_9_node_quadrangle, save_element)
    msh_handler.parse()
    
    nodes    = np.array(nodes)
    elements = np.array(elements)
    
    return nodes, elements


def plot_element_grid(x_nodes, y_nodes, xi_LGL, eta_LGL, axes_handler,
                      grid_width = 1., grid_color = 'red'):
    '''
    Uses the :math:`\\xi_{LGL}` and :math:`\\eta_{LGL}` points to plot a grid
    in the :math:`x-y` plane using the points corresponding to the
    :math:`(\\xi_{LGL}, \\eta_{LGL})` points.
    
    **Usage**
    
    .. code-block:: python
       :linenos:
       
       # Plots a grid for an element using 8 LGL points
       
       N_LGL        = 8
       xi_LGL       = lagrange.LGL_points(N)
       eta_LGL      = lagrange.LGL_points(N)
       
       # 8 x_nodes and y_nodes of an element
       x_nodes = [0., 0., 0., 0.5, 1., 1., 1., 0.5]
       y_nodes = [1., 0.5, 0., 0., 0., 0.5,  1., 1.]
       
       axes_handler = pyplot.axes()
       msh_parser.plot_element_grid(x_nodes, y_nodes,
                                    xi_LGL, eta_LGL, axes_handler)
       
       pyplot.title(r'Gird plot of an element.')
       pyplot.xlabel(r'$x$')
       pyplot.ylabel(r'$y$')
       
       pyplot.xlim(-.1, 1.1)
       pyplot.ylim(-.1, 1.1)
       
       pyplot.show()
    
    Parameters
    ----------
    
    x_nodes      : np.array [8]
                   x_nodes of the element.

    y_nodes      : np.array [8]
                   y_nodes of the element.

    xi_LGL       : np.array [N_LGL]
                   LGL points on the :math:`\\xi` axis

    eta_LGL      : np.array [N_LGL]
                   LGL points on the :math:`\\eta` axis

    axes_handler : matplotlib.axes.Axes
                   The plot handler being used to plot the element grid.
                   You may generate it by calling the function pyplot.axes()

    grid_width   : float
                   Grid line width.

    grid_color   : str
                   Grid line color.

    Returns
    -------

    None
    '''

    axes_handler.set_aspect('equal')

    N = xi_LGL.shape[0]

    xy_map = np.ndarray ((N, N, 2), float)

    for m in np.arange (N):
        for n in np.arange (N):
            xy_map[m][n][0] = isoparam.isoparam_x(x_nodes, xi_LGL[m], eta_LGL[n])
            xy_map[m][n][1] = isoparam.isoparam_y(y_nodes, xi_LGL[m], eta_LGL[n])

    array3d = xy_map.copy()
    N = array3d.shape[0]
    #Plot the vertical lines
    for m in np.arange (0, N):
        for n in np.arange (1, N):
            line = [array3d[m][n].tolist(), array3d[m][n-1].tolist()]
            (line1_xs, line1_ys) = zip(*line)
            axes_handler.add_line(lines.Line2D(line1_xs, line1_ys, linewidth=grid_width, color=grid_color))

    #Plot the horizontal lines
    for n in np.arange (0, N):
        for m in np.arange (1, N):
            line = [array3d[m][n].tolist(), array3d[m-1][n].tolist()]
            (line1_xs, line1_ys) = zip(*line)
            axes_handler.add_line(lines.Line2D(line1_xs, line1_ys, linewidth=grid_width, color=grid_color))

    return


def plot_element_boundary(x_nodes, y_nodes, axes_handler,
                          grid_width = 2., grid_color = 'blue'):
    '''
    Plots the boundary of a given :math:`2^{nd}` order element.
    
    Parameters
    ----------
    
    x_nodes      : np.ndarray [8]
                   :math:`x` nodes of the element.
                  
    y_nodes      : np.ndarray [8]
                   :math:`y` nodes of the element.
             
    axes_handler : matplotlib.axes.Axes
                   The plot handler being used to plot the element grid.
                   You may generate it by calling the function pyplot.axes()
                   
    grid_width   : float
                   Grid line width.
                 
    grid_color   : str
                   Grid line color.

    Returns
    -------
    
    None

    '''
    
    xi  = np.linspace(-1, 1, 20)
    eta = np.linspace(-1, 1, 20)
    
    left_edge   = np.zeros([xi.size, 2])
    bottom_edge = np.zeros([xi.size, 2])
    right_edge  = np.zeros([xi.size, 2])
    top_edge    = np.zeros([xi.size, 2])
    
    left_edge[:, 0]   = isoparam.isoparam_x(x_nodes, -1., eta)
    bottom_edge[:, 0] = isoparam.isoparam_x(x_nodes, xi, -1)
    right_edge[:, 0]  = isoparam.isoparam_x(x_nodes, 1., eta)
    top_edge[:, 0]    = isoparam.isoparam_x(x_nodes, xi, 1.)
    
    left_edge[:, 1]   = isoparam.isoparam_y(y_nodes, -1., eta)
    bottom_edge[:, 1] = isoparam.isoparam_y(y_nodes, xi, -1)
    right_edge[:, 1]  = isoparam.isoparam_y(y_nodes, 1., eta)
    top_edge[:, 1]    = isoparam.isoparam_y(y_nodes, xi, 1.)
    
    # Plot edges
    utils.plot_line(left_edge, axes_handler, grid_width, grid_color)
    utils.plot_line(bottom_edge, axes_handler, grid_width, grid_color)
    utils.plot_line(right_edge, axes_handler, grid_width, grid_color)
    utils.plot_line(top_edge, axes_handler, grid_width, grid_color)
    
    return

def plot_mesh_grid(nodes, elements, xi_LGL, eta_LGL, axes_handler):
    '''
    Plots the mesh grid.
    
    Parameters
    ----------
    
    nodes        : np.ndarray [N, 2]
                   Array of nodes in the mesh. First column and the second column are
                   the :math:`x` and :math:`y` coordinates respectivily.
                 
    elements     : np.ndarray [N_e, 8]
                   Array of elements.
                   
    xi_LGL       : np.array [N_LGL]
                   LGL points on the :math:`\\xi` axis
                 
    eta_LGL      : np.array [N_LGL]
                   LGL points on the :math:`\\eta` axis
    
    axes_handler : matplotlib.axes.Axes
                   The plot handler being used to plot the element grid.
                   You may generate it by calling the function pyplot.axes()

    Returns
    -------
    
    None
    '''

    for element in elements:
        msh_parser.plot_element_grid(nodes[element, 0], nodes[element, 1],
                                    xi_LGL, eta_LGL, axes_handler)
        msh_parser.plot_element_boundary(nodes[element, 0], nodes[element, 1],
                                        axes_handler)
    
    return
