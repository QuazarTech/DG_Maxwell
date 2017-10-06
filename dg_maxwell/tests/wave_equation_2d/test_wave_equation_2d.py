#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
sys.path.insert(0, os.path.abspath('./'))
import csv

import arrayfire as af
af.set_backend('cpu')
import numpy as np

from dg_maxwell import utils
from dg_maxwell import msh_parser
from dg_maxwell import lagrange
from dg_maxwell import wave_equation_2d

def test_dx_dxi():
    '''
    This test checks the derivative :math:`\\frac{\\partial x}{\\partial \\xi}`
    calculated using the function :meth:`dg_maxwell.wave_equation_2d.dx_dxi`
    for the :math:`0^{th}` element of a mesh for a circular ring. You may
    download the file from this
    :download:`link <dg_maxwell/tests/wave_equation_2d/files/circle.msh>`.
    '''
    threshold = 1e-8
    
    dx_dxi_reference = utils.csv_to_numpy(
        'dg_maxwell/tests/wave_equation_2d/files/dx_dxi_data.csv')
    
    nodes, elements = msh_parser.read_order_2_msh(
        'dg_maxwell/tests/wave_equation_2d/files/circle.msh')
    
    
    N_LGL = 16
    xi_LGL  = np.array(lagrange.LGL_points(N_LGL))
    eta_LGL = np.array(lagrange.LGL_points(N_LGL))
    Xi, Eta = np.meshgrid(xi_LGL, eta_LGL)
    
    dx_dxi = wave_equation_2d.dx_dxi(nodes[elements[0]][:, 0],
                                     Xi, Eta)
    
    assert np.all(np.isclose(dx_dxi, dx_dxi_reference))
    
    
def test_dx_deta():
    '''
    This test checks the derivative :math:`\\frac{\\partial x}{\\partial \\eta}`
    calculated using the function :meth:`dg_maxwell.wave_equation_2d.dx_deta`
    for the :math:`0^{th}` element of a mesh for a circular ring. You may
    download the file from this
    :download:`link <dg_maxwell/tests/wave_equation_2d/files/circle.msh>`.
    '''
    
    threshold = 1e-8
    
    dx_deta_reference = utils.csv_to_numpy(
        'dg_maxwell/tests/wave_equation_2d/files/dx_deta_data.csv')
    
    nodes, elements = msh_parser.read_order_2_msh(
        'dg_maxwell/tests/wave_equation_2d/files/circle.msh')
    
    
    N_LGL = 16
    xi_LGL  = np.array(lagrange.LGL_points(N_LGL))
    eta_LGL = np.array(lagrange.LGL_points(N_LGL))
    Xi, Eta = np.meshgrid(xi_LGL, eta_LGL)
    
    dx_deta = wave_equation_2d.dx_deta(nodes[elements[0]][:, 0],
                                       Xi, Eta)
    
    assert np.all(np.isclose(dx_deta, dx_deta_reference))
    
    
def test_dy_dxi():
    '''
    This test checks the derivative :math:`\\frac{\\partial y}{\\partial \\xi}`
    calculated using the function :meth:`dg_maxwell.wave_equation_2d.dy_dxi`
    for the :math:`0^{th}` element of a mesh for a circular ring. You may
    download the file from this
    :download:`link <dg_maxwell/tests/wave_equation_2d/files/circle.msh>`.
    '''
    threshold = 1e-8
    
    dy_dxi_reference = utils.csv_to_numpy(
        'dg_maxwell/tests/wave_equation_2d/files/dy_dxi_data.csv')
    
    nodes, elements = msh_parser.read_order_2_msh(
        'dg_maxwell/tests/wave_equation_2d/files/circle.msh')
    
    
    N_LGL = 16
    xi_LGL  = np.array(lagrange.LGL_points(N_LGL))
    eta_LGL = np.array(lagrange.LGL_points(N_LGL))
    Xi, Eta = np.meshgrid(xi_LGL, eta_LGL)
    
    dy_dxi = wave_equation_2d.dy_dxi(nodes[elements[0]][:, 1],
                                     Xi, Eta)
    
    assert np.all(np.isclose(dy_dxi, dy_dxi_reference))

    
def test_dy_deta():
    '''
    This test checks the derivative :math:`\\frac{\\partial y}{\\partial \\eta}`
    calculated using the function :meth:`dg_maxwell.wave_equation_2d.dy_deta`
    for the :math:`0^{th}` element of a mesh for a circular ring. You may
    download the file from this
    :download:`link <dg_maxwell/tests/wave_equation_2d/files/circle.msh>`.
    '''
    
    threshold = 1e-8
    
    dy_deta_reference = utils.csv_to_numpy(
        'dg_maxwell/tests/wave_equation_2d/files/dy_deta_data.csv')
    
    nodes, elements = msh_parser.read_order_2_msh(
        'dg_maxwell/tests/wave_equation_2d/files/circle.msh')
    
    
    N_LGL = 16
    xi_LGL  = np.array(lagrange.LGL_points(N_LGL))
    eta_LGL = np.array(lagrange.LGL_points(N_LGL))
    Xi, Eta = np.meshgrid(xi_LGL, eta_LGL)
    
    dy_deta = wave_equation_2d.dy_deta(nodes[elements[0]][:, 1],
                                       Xi, Eta)
    
    assert np.all(np.isclose(dy_deta, dy_deta_reference))
    