#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.lines as lines
import gmshtranslator.gmshtranslator as gmsh

import lib.isoparam as isoparam

plt.rcParams['figure.figsize']  = 12, 7.5
plt.rcParams['lines.linewidth'] = 1.5
plt.rcParams['font.family']     = 'serif'
plt.rcParams['font.weight']     = 'bold'
plt.rcParams['font.size']       = 20  
plt.rcParams['font.sans-serif'] = 'serif'
plt.rcParams['text.usetex']     = True
plt.rcParams['axes.linewidth']  = 1.5
plt.rcParams['axes.titlesize']  = 'medium'
plt.rcParams['axes.labelsize']  = 'medium'

plt.rcParams['xtick.major.size'] = 8
plt.rcParams['xtick.minor.size'] = 4
plt.rcParams['xtick.major.pad']  = 8
plt.rcParams['xtick.minor.pad']  = 8
plt.rcParams['xtick.color']      = 'k'
plt.rcParams['xtick.labelsize']  = 'medium'
plt.rcParams['xtick.direction']  = 'in'    

plt.rcParams['ytick.major.size'] = 8
plt.rcParams['ytick.minor.size'] = 4
plt.rcParams['ytick.major.pad']  = 8
plt.rcParams['ytick.minor.pad']  = 8
plt.rcParams['ytick.color']      = 'k'
plt.rcParams['ytick.labelsize']  = 'medium'
plt.rcParams['ytick.direction']  = 'in'
plt.rcParams['text.usetex'] = True
plt.rcParams['text.latex.unicode'] = True

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
             Nodes is a :math:`N \\times 2` array, where, :math:`N` is the total
             number of Nodes in the mesh. Each node contains it's :math:`(x, y)`
             coordinates.
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
        print ("Node ", tag, x, y, z)
        nodes.append([x, y])
        return

    
    def is_9_node_quadrangle (eletag, eletype,
                              physgrp, nodes):
        return eletype == msh_handler.quadrangle_9_node

    def save_element (eletag, eletype,
                      physgrp, node_tags):
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


def plot_grid(x_nodes, y_nodes, xi_LGL, eta_LGL, axes_handler, grid_width = 1., grid_color = 'red'):
    '''
    '''
    
    axes_handler.set_aspect('equal')
    
    N = xi_LGL.shape[0]
    
    xy_map = np.ndarray ((N, N, 2), float)
    
    for m in np.arange (N):
        for n in np.arange (N):
            xy_map[m][n][0] = isoparam.isoparam_x(x_nodes, xi_LGL[m], eta_LGL[n])
            xy_map[m][n][1] = isoparam.isoparam_y(y_nodes, xi_LGL[m], eta_LGL[n])
            pass
        pass
    
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
