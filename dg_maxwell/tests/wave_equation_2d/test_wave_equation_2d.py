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
from dg_maxwell import global_variables as gvar

af.set_backend(params.backend)
af.set_device(params.device)




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
    advec_var = gvar.advection_variables(params.N_LGL, params.N_quad,
                                         params.x_nodes, params.N_Elements,
                                         params.c, params.total_time,
                                         params.wave, params.c_x, params.c_y,
                                         params.courant, params.mesh_file,
                                         params.total_time_2d)
    
    A_matrix_test = wave_equation_2d.A_matrix(params.N_LGL, advec_var)
    
    assert af.max(af.abs(A_matrix_test - A_matrix_ref)) < threshold


def test_interpolation():
    '''
    '''
    threshold = 8e-9
    params.N_LGL = 8

    gv = gvar.advection_variables(params.N_LGL, params.N_quad,\
                                          params.x_nodes, params.N_Elements,\
                                          params.c, params.total_time, params.wave,\
                                          params.c_x, params.c_y, params.courant,\
                                          params.mesh_file, params.total_time_2d)


    N_LGL = 8
    xi_LGL = lagrange.LGL_points(N_LGL)
    xi_i  = af.flat(af.transpose(af.tile(xi_LGL, 1, N_LGL)))
    eta_j = af.tile(xi_LGL, N_LGL)
    f_ij  = np.e ** (xi_i + eta_j)
    interpolated_f = wave_equation_2d.lag_interpolation_2d(f_ij, gv.Li_Lj_coeffs)
    xi  = utils.linspace(-1, 1, 8)
    eta = utils.linspace(-1, 1, 8)

    assert (af.mean(af.transpose(utils.polyval_2d(interpolated_f, xi, eta))
                    - np.e**(xi+eta)) < threshold)



def test_Li_Lj_coeffs():
    '''
    '''
    threshold = 2e-9
    
    N_LGL = 8
    
    numerical_L3_xi_L4_eta_coeffs = wave_equation_2d.Li_Lj_coeffs(N_LGL)[:, :, 28]

    analytical_L3_xi_L4_eta_coeffs = af.np_to_af_array(np.array([\
            [-129.727857225405, 27.1519390573796, 273.730966722451, - 57.2916772505673\
            , - 178.518337439857, 37.3637484073274, 34.5152279428116, -7.22401021413973], \
            [- 27.1519390573797, 5.68287960923199, 57.2916772505669, - 11.9911032408375,\
            - 37.3637484073272, 7.82020331954072, 7.22401021413968, - 1.51197968793550 ],\
            [273.730966722451, - 57.2916772505680,- 577.583286622990, 120.887730163458,\
            376.680831166362, - 78.8390033617950, - 72.8285112658236, 15.2429504489039],\
            [57.2916772505673, - 11.9911032408381, - 120.887730163459, 25.3017073771593, \
            78.8390033617947, -16.5009417437969, - 15.2429504489039, 3.19033760747451],\
            [- 178.518337439857, 37.3637484073272, 376.680831166362, - 78.8390033617954,\
            - 245.658854496594, 51.4162061168383, 47.4963607700889, - 9.94095116237084],\
            [- 37.3637484073274, 7.82020331954070, 78.8390033617948, - 16.5009417437970,\
            - 51.4162061168385, 10.7613717277423, 9.94095116237085, -2.08063330348620],\
            [34.5152279428116, - 7.22401021413972, - 72.8285112658235, 15.2429504489038,\
            47.4963607700889, - 9.94095116237085, - 9.18307744707700, 1.92201092760671],\
            [7.22401021413973, - 1.51197968793550, -15.2429504489039, 3.19033760747451,\
            9.94095116237084, - 2.08063330348620, - 1.92201092760671, 0.402275383947182]]))

    af.display(numerical_L3_xi_L4_eta_coeffs - analytical_L3_xi_L4_eta_coeffs, 14)
    assert (af.max(af.abs(numerical_L3_xi_L4_eta_coeffs - analytical_L3_xi_L4_eta_coeffs)) <= threshold)

