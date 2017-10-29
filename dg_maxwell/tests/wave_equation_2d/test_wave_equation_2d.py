#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
sys.path.insert(0, os.path.abspath('./'))
import csv

import numpy as np
import arrayfire as af

from dg_maxwell import params
from dg_maxwell import utils
from dg_maxwell import msh_parser
from dg_maxwell import lagrange
from dg_maxwell import wave_equation_2d

af.set_backend(params.backend)

def test_dx_dxi():
    '''
    This test checks the derivative :math:`\\frac{\\partial x}{\\partial \\xi}`
    calculated using the function :math:`dg_maxwell.wave_equation_2d.dx_dxi`
    for the :math:`0^{th}` element of a mesh for a circular ring. You may
    download the file from this
    :download:`link <../dg_maxwell/tests/wave_equation_2d/files/circle.msh>`.
    '''
    threshold = 1e-7
    
    dx_dxi_reference = af.np_to_af_array(utils.csv_to_numpy(
        'dg_maxwell/tests/wave_equation_2d/files/dx_dxi_data.csv'))
    
    nodes, elements = msh_parser.read_order_2_msh(
        'dg_maxwell/tests/wave_equation_2d/files/circle.msh')
    
    N_LGL   = 16
    xi_LGL  = lagrange.LGL_points(N_LGL)
    eta_LGL = lagrange.LGL_points(N_LGL)

    Xi  = af.data.tile(af.array.transpose(xi_LGL), d0 = N_LGL)
    Eta = af.data.tile(eta_LGL, d0 = 1, d1 = N_LGL)

    dx_dxi  = wave_equation_2d.dx_dxi(nodes[elements[0]][:, 0],
                                      Xi, Eta)

    check = af.abs((dx_dxi - dx_dxi_reference)) < threshold

    assert af.all_true(check)
    
    
def test_dx_deta():
    '''
    This test checks the derivative :math:`\\frac{\\partial x}{\\partial \\eta}`
    calculated using the function :math:`dg_maxwell.wave_equation_2d.dx_deta`
    for the :math:`0^{th}` element of a mesh for a circular ring. You may
    download the file from this
    :download:`link <../dg_maxwell/tests/wave_equation_2d/files/circle.msh>`.
    '''
    
    threshold = 1e-7
    
    dx_deta_reference = af.np_to_af_array(utils.csv_to_numpy(
        'dg_maxwell/tests/wave_equation_2d/files/dx_deta_data.csv'))
    
    nodes, elements = msh_parser.read_order_2_msh(
        'dg_maxwell/tests/wave_equation_2d/files/circle.msh')
    
    
    N_LGL   = 16
    xi_LGL  = lagrange.LGL_points(N_LGL)
    eta_LGL = lagrange.LGL_points(N_LGL)
    Xi  = af.data.tile(af.array.transpose(xi_LGL), d0 = N_LGL)
    Eta = af.data.tile(eta_LGL, d0 = 1, d1 = N_LGL)
    
    dx_deta = wave_equation_2d.dx_deta(nodes[elements[0]][:, 0],
                                       Xi, Eta)
    
    check = af.abs(dx_deta - dx_deta_reference) < threshold
    
    assert af.all_true(check)
    
    
def test_dy_dxi():
    '''
    This test checks the derivative :math:`\\frac{\\partial y}{\\partial \\xi}`
    calculated using the function :math:`dg_maxwell.wave_equation_2d.dy_dxi`
    for the :math:`0^{th}` element of a mesh for a circular ring. You may
    download the file from this
    :download:`link <../dg_maxwell/tests/wave_equation_2d/files/circle.msh>`.
    '''
    threshold = 1e-7
    
    dy_dxi_reference = af.np_to_af_array(utils.csv_to_numpy(
        'dg_maxwell/tests/wave_equation_2d/files/dy_dxi_data.csv'))
    
    nodes, elements = msh_parser.read_order_2_msh(
        'dg_maxwell/tests/wave_equation_2d/files/circle.msh')
    
    
    N_LGL   = 16
    xi_LGL  = lagrange.LGL_points(N_LGL)
    eta_LGL = lagrange.LGL_points(N_LGL)
    Xi  = af.data.tile(af.array.transpose(xi_LGL), d0 = N_LGL)
    Eta = af.data.tile(eta_LGL, d0 = 1, d1 = N_LGL)
    
    dy_dxi  = wave_equation_2d.dy_dxi(nodes[elements[0]][:, 1],
                                      Xi, Eta)
    
    check = af.abs(dy_dxi - dy_dxi_reference) < threshold
    
    assert af.all_true(check)

    
def test_dy_deta():
    '''
    This test checks the derivative :math:`\\frac{\\partial y}{\\partial \\eta}`
    calculated using the function :math:`dg_maxwell.wave_equation_2d.dy_deta`
    for the :math:`0^{th}` element of a mesh for a circular ring. You may
    download the file from this
    :download:`link <../dg_maxwell/tests/wave_equation_2d/files/circle.msh>`.
    '''
    
    threshold = 1e-7
    
    dy_deta_reference = af.np_to_af_array(utils.csv_to_numpy(
        'dg_maxwell/tests/wave_equation_2d/files/dy_deta_data.csv'))
    
    nodes, elements = msh_parser.read_order_2_msh(
        'dg_maxwell/tests/wave_equation_2d/files/circle.msh')
    
    
    N_LGL   = 16
    xi_LGL  = lagrange.LGL_points(N_LGL)
    eta_LGL = lagrange.LGL_points(N_LGL)
    Xi  = af.data.tile(af.array.transpose(xi_LGL), d0 = N_LGL)
    Eta = af.data.tile(eta_LGL, d0 = 1, d1 = N_LGL)
    
    dy_deta = wave_equation_2d.dy_deta(nodes[elements[0]][:, 1],
                                       Xi, Eta)
    
    check = af.abs(dy_deta - dy_deta_reference) < threshold
    
    assert af.all_true(check)


def test_jacobian():
    '''
    This test checks the derivative Jacobian
    calculated using the function :math:`dg_maxwell.wave_equation_2d.jacobian`
    for the :math:`0^{th}` element of a mesh for a circular ring. You may
    download the file from this
    :download:`link <../dg_maxwell/tests/wave_equation_2d/files/circle.msh>`.
    '''
    
    threshold = 1e-7
    
    jacobian_reference = af.np_to_af_array(utils.csv_to_numpy(
        'dg_maxwell/tests/wave_equation_2d/files/jacobian_data.csv'))
    
    nodes, elements = msh_parser.read_order_2_msh(
        'dg_maxwell/tests/wave_equation_2d/files/circle.msh')
    
    
    N_LGL   = 16
    xi_LGL  = lagrange.LGL_points(N_LGL)
    eta_LGL = lagrange.LGL_points(N_LGL)
    Xi  = af.data.tile(af.array.transpose(xi_LGL), d0 = N_LGL)
    Eta = af.data.tile(eta_LGL, d0 = 1, d1 = N_LGL)
    
    jacobian = wave_equation_2d.jacobian(nodes[elements[0]][:, 0],
                                         nodes[elements[0]][:, 1],
                                         Xi, Eta)
    
    check = af.abs(jacobian - jacobian_reference) < threshold
    
    assert af.all_true(check)


def test_A_matrix():
    '''
    Compares the tensor product calculated using the ``A_matrix`` function
    with an analytic value of the tensor product for :math:`N_{LGL} = 4`.
    The analytic value of the tensor product is calculated in this
    `document`_
    
    .. _document: https://goo.gl/QNWxXp
    '''
    threshold = 1e-12
    
    A_matrix_ref = af.np_to_af_array(utils.csv_to_numpy(
        'dg_maxwell/tests/wave_equation_2d/files/A_matrix_ref.csv'))
    
    params.N_LGL = 4
    
    A_matrix_test = wave_equation_2d.A_matrix()
    
    assert af.all_true(af.abs(A_matrix_test - A_matrix_ref) < threshold)
