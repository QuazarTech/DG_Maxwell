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
    print(A_matrix_ref, A_matrix_test)
    
    assert af.max(af.abs(A_matrix_test - A_matrix_ref)) < threshold


def test_interpolation():
    '''
    '''
    threshold = 8e-8
    
    N_LGL = 8
    xi_LGL = lagrange.LGL_points(N_LGL)
    xi_i  = af.flat(af.transpose(af.tile(xi_LGL, 1, N_LGL)))
    eta_j = af.tile(xi_LGL, N_LGL)
    f_ij  = np.e ** (xi_i + eta_j)
    interpolated_f = wave_equation_2d.lag_interpolation_2d(f_ij, N_LGL)
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

def test_surface_term_2d():
    '''
    link to sage worksheet https://goo.gl/bvcjBG
    '''
    threshold = 1e-11
    
    params.N_LGL = 16
    analytical_surface_term_xi_1_boundary = af.np_to_af_array(np.array(
                                        [0.000518137698965616, 0.00316169869369534, 0.00555818937089650,\
                                         0.00772576775028549, 0.00957686227446707, 0.0110358302160314,\
                                         0.0120429724206691, 0.0125570655998897, 0.0125570656010149,\
                                         0.0120429724194322, 0.0110358302175448, 0.00957686227250284,\
                                         0.00772576775253881 , 0.00555818936935162 , 0.00316169869299086,\
                                         0.000518137700074415 ]))

    analytical_surface_term_xi_minus1_boundary = af.np_to_af_array(np.array(
                                             [0.000518137699431318, 0.00316169869653708, 0.00555818937589221,\
                                              0.00772576775722942, 0.00957686228307477, 0.0110358302259504, \
                                              0.0120429724314933, 0.0125570656111760, 0.0125570656123012, \
                                              0.0120429724302564, 0.0110358302274638, 0.00957686228111053, \
                                              0.00772576775948273, 0.00555818937434732, 0.00316169869583259, \
                                              0.000518137700540118]))
    xi_LGL = lagrange.LGL_points(params.N_LGL)
    xi_i   = af.flat(af.transpose(af.tile(xi_LGL, 1, params.N_LGL)))
    eta_j  = af.tile(xi_LGL, params.N_LGL)

    u_init_2d = np.e ** (- (xi_i ** 2) / (0.6 ** 2))
    print(u_init_2d.shape)

    numerical_surface_term = wave_equation_2d.surface_term(u_init_2d)
    print(af.mean(af.abs(numerical_surface_term[params.N_LGL:-params.N_LGL])))

    error = af.mean(af.abs(numerical_surface_term[-params.N_LGL:] - analytical_surface_term_xi_1_boundary))\
           +af.mean(af.abs(numerical_surface_term[:params.N_LGL] - analytical_surface_term_xi_minus1_boundary))

    assert error<= threshold
