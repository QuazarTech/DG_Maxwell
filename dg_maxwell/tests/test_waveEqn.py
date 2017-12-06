#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
sys.path.insert(0, os.path.abspath('./'))

import numpy as np
import arrayfire as af
from matplotlib import pyplot as pl

from dg_maxwell import params
from dg_maxwell import lagrange
from dg_maxwell import wave_equation
from dg_maxwell import wave_equation_2d
from dg_maxwell import isoparam
from dg_maxwell import global_variables
from dg_maxwell import utils

af.set_backend(params.backend)
af.set_device(params.device)
# This test uses the initial paramters N_LGL = 8, N_Elements = 10 and c = 1.


def test_A_matrix():
    '''
    Test function to check the A matrix obtained from wave_equation module with
    one obtained by numerical integral solvers.
    '''
    threshold = 1e-8


    params.N_LGL      = 8
    params.N_quad     = 10
    params.N_Elements = 10
    wave              = 'gaussian'

    gv = global_variables.advection_variables(params.N_LGL, params.N_quad,\
                                          params.x_nodes, params.N_Elements,\
                                          params.c, params.total_time, wave,\
                                          params.c_x, params.c_y, params.courant,\
                                          params.mesh_file, params.total_time_2d)

    
    
    reference_A_matrix = 0.1 * af.interop.np_to_af_array(np.array([\

    [0.03333333333332194, 0.005783175201965206, -0.007358427761753982, \
    0.008091331778355441, -0.008091331778233877, 0.007358427761705623, \
    -0.00578317520224949, 0.002380952380963754], \
    
    [0.005783175201965206, 0.19665727866729804, 0.017873132323192046,\
    -0.01965330750343234, 0.019653307503020866, -0.017873132322725152,\
    0.014046948476303067, -0.005783175202249493], \
    
    [-0.007358427761753982, 0.017873132323192046, 0.31838117965137114, \
    0.025006581762566073, -0.025006581761945083, 0.022741512832051156,\
    -0.017873132322725152, 0.007358427761705624], \
    
    [0.008091331778355441, -0.01965330750343234, 0.025006581762566073, \
    0.3849615416814164, 0.027497252976343693, -0.025006581761945083, \
    0.019653307503020863, -0.008091331778233875],\
    
    [-0.008091331778233877, 0.019653307503020866, -0.025006581761945083, \
    0.027497252976343693, 0.3849615416814164, 0.025006581762566073, \
    -0.019653307503432346, 0.008091331778355443], \
    
    [0.007358427761705623, -0.017873132322725152, 0.022741512832051156, \
    -0.025006581761945083, 0.025006581762566073, 0.31838117965137114, \
    0.017873132323192046, -0.0073584277617539835], \
    
    [-0.005783175202249493, 0.014046948476303067, -0.017873132322725152, \
    0.019653307503020863, -0.019653307503432346, 0.017873132323192046, \
    0.19665727866729804, 0.0057831752019652065], \
    
    [0.002380952380963754, -0.005783175202249493, 0.007358427761705624, \
    -0.008091331778233875, 0.008091331778355443, -0.0073584277617539835, \
    0.0057831752019652065, 0.033333333333321946]

    ]))
    
    test_A_matrix = wave_equation.A_matrix(gv)
    print(test_A_matrix, reference_A_matrix)
    error_array = af.abs(reference_A_matrix - test_A_matrix)
    
    assert af.max(error_array) < threshold


def test_LGL_points():
    '''
    Comparing the LGL nodes obtained by LGL_points with
    the reference nodes for N = 6
    '''
    threshold = 5e-13
    reference_nodes  = \
        af.np_to_af_array(np.array([-1.,                 -0.7650553239294647,\
                                    -0.28523151648064504, 0.28523151648064504,\
                                     0.7650553239294647,  1. \
                                   ] \
                                  ) \
                         )

    calculated_nodes = (lagrange.LGL_points(6))
    assert(af.max(af.abs(reference_nodes - calculated_nodes)) <= threshold)


def test_gauss_nodes():
    '''
    The Gauss points obtained by the function above is compared to
    analytical values.
    
    **See:** `https://goo.gl/9gqLpe`
    '''
    threshold = 1e-10
    analytical_gauss_nodes = np.array([-0.906179845938664, -0.5384693101056831,\
                                        0, 0.5384693101056831, 0.906179845938664])
    calculated_gauss_nodes = lagrange.gauss_nodes(5)
    
    assert np.max(abs(analytical_gauss_nodes - calculated_gauss_nodes)) <= threshold


def test_gauss_weights():
    '''
    Test to check the gaussian weights calculated.
    '''
    threshold = 2e-8
    analytical_gauss_weights = af.Array([0.23692688505618908, 0.47862867049936647,\
                                         0.5688888888888889, 0.47862867049936647, \
                                         0.23692688505618908
                                        ]
                                       )
    calculated_gauss_weights = lagrange.gaussian_weights(5)

    assert af.max(af.abs(analytical_gauss_weights - calculated_gauss_weights))\
                                                                  <= threshold


def test_isoparam_1D():
    '''
    A test function to check the mapping_xi_to_x function in wave_equation module,
    The test involves passing trial element nodes and :math: `\\xi` and
    comparing it with the x obatined by passing the trial parameters to
    mapping_xi_to_x function.
    '''
    threshold = 1e-14
    
    test_element_nodes = af.interop.np_to_af_array(np.array([7, 11]))
    test_xi            = 0
    analytical_x_value = 9
    numerical_x_value  = isoparam.isoparam_1D(test_element_nodes, test_xi)
    
    assert af.abs(analytical_x_value - numerical_x_value) <= threshold


def test_dx_dxi():
    '''
    A Test function to check the dx_xi function in wave_equation module by
    passing nodes of an element and using the LGL points. Analytically, the
    differential would be a constant. The check has a tolerance 1e-7.
    '''
    threshold = 1e-8

    params.N_LGL      = 8
    params.N_quad     = 10
    params.N_Elements = 10
    wave              = 'gaussian'

    gv = global_variables.advection_variables(params.N_LGL, params.N_quad,\
                                          params.x_nodes, params.N_Elements,\
                                          params.c, params.total_time, wave,\
                                          params.c_x, params.c_y, params.courant,\
                                          params.mesh_file, params.total_time_2d)



    nodes = np.array([7, 10], dtype = np.float64)
    test_nodes = af.interop.np_to_af_array(nodes)
    analytical_dx_dxi = 1.5
    check_dx_dxi = abs((af.mean(wave_equation.dx_dxi_numerical
                    (test_nodes,gv.xi_LGL)) - analytical_dx_dxi)) <= threshold
    
    assert check_dx_dxi


def test_dx_dxi_analytical():
    '''
    Test to check the dx_dxi_analytical in wave equation module for an element
    and compare it with an analytical value.
    '''
    threshold = 1e-14

    nodes = af.Array([2,6])
    check_analytical_dx_dxi = af.sum(af.abs(wave_equation.dx_dxi_analytical
                                         (nodes, 0) - 2)) <= threshold
    assert check_analytical_dx_dxi


def test_LGL_points():
    '''
    Comparing the LGL nodes obtained by LGL_points with
    the reference nodes for N = 6
    '''
    threshold = 5e-13
    reference_nodes  = \
        af.np_to_af_array(np.array([-1.,                 -0.7650553239294647,\
                                    -0.28523151648064504, 0.28523151648064504,\
                                     0.7650553239294647,  1. \
                                   ] \
                                  ) \
                         )

    calculated_nodes = (lagrange.LGL_points(6))
    assert(af.max(af.abs(reference_nodes - calculated_nodes)) <= threshold)

def test_lagrange_coeffs():
    '''
    Function to test the lagrange_coeffs in global_variables module by
    passing 8 LGL points and comparing the numerically obtained basis function
    coefficients to analytically calculated ones.
    
    Reference
    ---------
    The link to the sage worksheet where the calculations were carried out.
    
    `https://goo.gl/6EFX5S`
    '''
    threshold = 1e-10

    params.N_LGL      = 8
    params.N_quad     = 10
    params.N_Elements = 10
    wave              = 'gaussian'

    gv = global_variables.advection_variables(params.N_LGL, params.N_quad,\
                                          params.x_nodes, params.N_Elements,\
                                          params.c, params.total_time, wave,\
                                          params.c_x, params.c_y, params.courant,\
                                          params.mesh_file, params.total_time_2d)

    basis_array_analytical = np.zeros([8, 8])
    
    basis_array_analytical[0] = np.array([-3.351562500008004,\
                                        3.351562500008006, \
                                        3.867187500010295,\
                                        -3.867187500010297,\
                                        - 1.054687500002225, \
                                        1.054687500002225, \
                                        0.03906249999993106,\
                                        - 0.03906249999993102])
    basis_array_analytical[1] = np.array([8.140722718246403,\
                                        - 7.096594831382852,\
                                        - 11.34747768400062,\
                                        9.89205188146461, \
                                        3.331608712119162, \
                                        - 2.904297073479968,\
                                        - 0.1248537463649464,\
                                        0.1088400233982081])
    basis_array_analytical[2] = np.array([-10.35813682892759,\
                                        6.128911440984293,\
                                        18.68335515838398,\
                                        - 11.05494463699297,\
                                        - 8.670037141196786,\
                                        5.130062549476987,\
                                        0.3448188117404021,\
                                        - 0.2040293534683072])

    basis_array_analytical[3] = np.array([11.38981374849497,\
                                        - 2.383879109609436,\
                                        - 24.03296250200938,\
                                        5.030080255538657,\
                                        15.67350804691132,\
                                        - 3.28045297599924,\
                                        - 3.030359293396907,\
                                        0.6342518300700298])

    basis_array_analytical[4] = np.array([-11.38981374849497,\
                                        - 2.383879109609437,\
                                        24.03296250200939,\
                                        5.030080255538648,\
                                        - 15.67350804691132,\
                                        - 3.28045297599924,\
                                        3.030359293396907,\
                                        0.6342518300700299])

    basis_array_analytical[5] = np.array([10.35813682892759,\
                                        6.128911440984293,\
                                        -18.68335515838398,\
                                        - 11.05494463699297,\
                                        8.670037141196786,\
                                        5.130062549476987,\
                                        - 0.3448188117404021,\
                                        - 0.2040293534683072])
    basis_array_analytical[6] = np.array([-8.140722718246403,\
                                        - 7.096594831382852,\
                                        11.34747768400062,\
                                        9.89205188146461, \
                                        -3.331608712119162, \
                                        - 2.904297073479968,\
                                        0.1248537463649464,\
                                        0.1088400233982081])
    basis_array_analytical[7] = np.array([3.351562500008004,\
                                        3.351562500008005, \
                                        - 3.867187500010295,\
                                        - 3.867187500010298,\
                                        1.054687500002225, \
                                        1.054687500002224, \
                                        - 0.039062499999931,\
                                        - 0.03906249999993102])
                
    basis_array_analytical = af.interop.np_to_af_array(basis_array_analytical)
    
    assert af.max(af.abs(basis_array_analytical - gv.lagrange_coeffs)) \
            < threshold


def test_volume_integral_flux():
    '''
    A test function to check the volume_integral_flux function in wave_equation
    module by analytically calculated Gauss-Lobatto quadrature.
    
    Reference
    ---------
    The link to the sage worksheet where the calculations were caried out is
    given below.
    `https://goo.gl/5Mub8M`
    '''
    threshold = 4e-9
    params.c = 1
    
    params.N_LGL      = 8
    params.N_quad     = 10
    params.N_Elements = 10
    wave              = 'gaussian'

    gv = global_variables.advection_variables(params.N_LGL, params.N_quad,\
                                          params.x_nodes, params.N_Elements,\
                                          params.c, params.total_time, wave,\
                                          params.c_x, params.c_y, params.courant,\
                                          params.mesh_file, params.total_time_2d)


    reference_flux_integral = af.transpose(af.interop.np_to_af_array(np.array
        ([
        [-0.002016634876668093, -0.000588597708116113, -0.0013016773719126333,\
        -0.002368387579324652, -0.003620502047659841, -0.004320197094090966,
        -0.003445512010153811, 0.0176615086879261],\

        [-0.018969769374, -0.00431252844519,-0.00882630935977,-0.0144355176966,\
        -0.019612124119, -0.0209837936827, -0.0154359890788, 0.102576031756], \

        [-0.108222418798, -0.0179274222595, -0.0337807018822, -0.0492589052599,\
        -0.0588472807471, -0.0557970236273, -0.0374764132459, 0.361310165819],\

        [-0.374448714304, -0.0399576371245, -0.0683852285846, -0.0869229749357,\
        -0.0884322503841, -0.0714664112839, -0.0422339853622, 0.771847201979], \

        [-0.785754362849, -0.0396035640187, -0.0579313769517, -0.0569022801117,\
        -0.0392041960688, -0.0172295769141, -0.00337464521455, 1.00000000213],\

        [-1.00000000213, 0.00337464521455, 0.0172295769141, 0.0392041960688,\
        0.0569022801117, 0.0579313769517, 0.0396035640187, 0.785754362849],\

        [-0.771847201979, 0.0422339853622, 0.0714664112839, 0.0884322503841, \
        0.0869229749357, 0.0683852285846, 0.0399576371245, 0.374448714304],\

        [-0.361310165819, 0.0374764132459, 0.0557970236273, 0.0588472807471,\
        0.0492589052599, 0.0337807018822, 0.0179274222595, 0.108222418798], \

        [-0.102576031756, 0.0154359890788, 0.0209837936827, 0.019612124119, \
        0.0144355176966, 0.00882630935977, 0.00431252844519, 0.018969769374],\

        [-0.0176615086879, 0.00344551201015 ,0.00432019709409, 0.00362050204766,\
        0.00236838757932, 0.00130167737191, 0.000588597708116, 0.00201663487667]\

         ])))
    
    numerical_flux = wave_equation.volume_integral_flux(gv.u_init[:, :], gv)
    assert (af.mean(af.abs(numerical_flux - reference_flux_integral)) < threshold)
    return numerical_flux

def test_lax_friedrichs_flux():
    '''
    A test function to test the lax_friedrichs_flux function in wave_equation
    module.
    '''
    threshold = 1e-14

    params.N_LGL      = 8
    params.N_quad     = 10
    params.N_Elements = 10
    wave              = 'gaussian'

    gv = global_variables.advection_variables(params.N_LGL, params.N_quad,\
                                          params.x_nodes, params.N_Elements,\
                                          params.c, params.total_time, wave,\
                                          params.c_x, params.c_y, params.courant,\
                                          params.mesh_file, params.total_time_2d)

    f_i = wave_equation.lax_friedrichs_flux(gv.u_init)
    analytical_lax_friedrichs_flux = gv.u_init[-1, :]

    assert af.max(af.abs(analytical_lax_friedrichs_flux - f_i)) < threshold


def test_surface_term():
    '''
    A test function to test the surface_term function in the wave_equation
    module using analytical Lax-Friedrichs flux.
    '''
    threshold = 1e-13
    params.c = 1

    params.N_LGL      = 8
    params.N_quad     = 10
    params.N_Elements = 10
    wave              = 'gaussian'

    gv = global_variables.advection_variables(params.N_LGL, params.N_quad,\
                                          params.x_nodes, params.N_Elements,\
                                          params.c, params.total_time, wave,\
                                          params.c_x, params.c_y, params.courant,\
                                          params.mesh_file, params.total_time_2d)


    analytical_f_i        = (gv.u_init[-1, :])
    analytical_f_i_minus1 = (af.shift(gv.u_init[-1, :], 0, 1))
    
    L_p_1                 = af.constant(0, params.N_LGL, dtype = af.Dtype.f64)
    L_p_1[params.N_LGL - 1] = 1
    
    L_p_minus1    = af.constant(0, params.N_LGL, dtype = af.Dtype.f64)
    L_p_minus1[0] = 1
    
    analytical_surface_term = af.blas.matmul(L_p_1, analytical_f_i)\
        - af.blas.matmul(L_p_minus1, analytical_f_i_minus1)
    
    numerical_surface_term = (wave_equation.surface_term(gv.u_init[:, :], gv))
    assert af.max(af.abs(analytical_surface_term - numerical_surface_term)) \
        < threshold
    return analytical_surface_term


def test_b_vector():
    '''
    A test function to check the b vector obtained analytically and compare it
    with the one returned by b_vector function in wave_equation module.
    '''
    threshold = 1e-13
    params.c = 1
    
    params.N_LGL      = 8
    params.N_quad     = 10
    params.N_Elements = 10
    wave              = 'gaussian'

    gv = global_variables.advection_variables(params.N_LGL, params.N_quad,\
                                          params.x_nodes, params.N_Elements,\
                                          params.c, params.total_time, wave,\
                                          params.c_x, params.c_y, params.courant,\
                                          params.mesh_file, params.total_time_2d)


    u_n_A_matrix         = af.blas.matmul(wave_equation.A_matrix(gv),\
                                                  gv.u_init)
    volume_integral_flux = test_volume_integral_flux()
    surface_term         = test_surface_term()
    b_vector_analytical  = u_n_A_matrix + (volume_integral_flux -\
                                    (surface_term)) * gv.delta_t
    b_vector_array       = wave_equation.b_vector(gv.u_init, gv)
    
    assert (b_vector_analytical - b_vector_array) < threshold

def test_integrate():
    '''
    Testing the integrate() function by passing coefficients
    of a polynomial and comparing it to the analytical result.
    '''
    threshold = 1e-14

    params.N_LGL      = 8
    params.N_quad     = 10
    params.N_Elements = 10
    wave              = 'gaussian'

    gv = global_variables.advection_variables(params.N_LGL, params.N_quad,\
                                          params.x_nodes, params.N_Elements,\
                                          params.c, params.total_time, wave,\
                                          params.c_x, params.c_y, params.courant,\
                                          params.mesh_file, params.total_time_2d)


    test_coeffs = af.np_to_af_array(np.array([7., 6, 4, 2, 1, 3, 9, 2]))
    # The coefficients of a test polynomial
    # `7x^7 + 6x^6 + 4x^5 + 2x^4 + x^3 + 3x^2 + 9x + 2`

    # Using integrate() function.

    calculated_integral = lagrange.integrate(af.transpose(test_coeffs), gv)

    analytical_integral = 8.514285714285714

    assert (calculated_integral - analytical_integral) <= threshold
