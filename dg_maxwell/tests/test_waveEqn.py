#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
sys.path.insert(0, os.path.abspath('./'))

import numpy as np
import arrayfire as af


from dg_maxwell import params
from dg_maxwell import lagrange
from dg_maxwell import wave_equation
from dg_maxwell import wave_equation_2d
from dg_maxwell import isoparam
from dg_maxwell import utils

af.set_backend(params.backend)

# This test uses the initial paramters N_LGL = 8, N_Elements = 10 and c = 1.



def change_parameters(LGL, Elements, quad, wave='sin'):
    '''

    Changes the parameters of the simulation. Used only for convergence tests.

    Parameters
    ----------
    LGL      : int
               The new N_LGL.

    Elements : int
               The new N_Elements.

    '''
    # The domain of the function.
    params.x_nodes    = af.np_to_af_array(np.array([-1., 1.]))

    # The number of LGL points into which an element is split.
    params.N_LGL      = LGL

    # Number of elements the domain is to be divided into.
    params.N_Elements = Elements

    # The number quadrature points to be used for integration.
    params.N_quad     = quad

    # Array containing the LGL points in xi space.
    params.xi_LGL     = lagrange.LGL_points(params.N_LGL)

    # N_Gauss number of Gauss nodes.
    params.gauss_points  = af.np_to_af_array(lagrange.gauss_nodes\
                                                    (params.N_quad))
    # The Gaussian weights.
    params.gauss_weights = lagrange.gaussian_weights(params.N_quad)

    # The lobatto nodes to be used for integration.
    params.lobatto_quadrature_nodes   = lagrange.LGL_points(params.N_quad)

    # The lobatto weights to be used for integration.
    params.lobatto_weights_quadrature = lagrange.lobatto_weights\
                                        (params.N_quad)

    # A list of the Lagrange polynomials in poly1d form.
    #params.lagrange_product = lagrange.product_lagrange_poly(params.xi_LGL)

    # An array containing the coefficients of the lagrange basis polynomials.
    params.lagrange_coeffs  = af.np_to_af_array(\
                              lagrange.lagrange_polynomials(params.xi_LGL)[1])

    # Refer corresponding functions.
    params.lagrange_basis_value = lagrange.lagrange_function_value\
                                           (params.lagrange_coeffs)


    # While evaluating the volume integral using N_LGL
    # lobatto quadrature points, The integration can be vectorized
    # and in this case the coefficients of the differential of the
    # Lagrange polynomials is required
    params.diff_pow      = (af.flip(af.transpose(af.range(params.N_LGL - 1) + 1), 1))
    params.dl_dxi_coeffs = (af.broadcast(utils.multiply, params.lagrange_coeffs[:, :-1], params.diff_pow))



    # Obtaining an array consisting of the LGL points mapped onto the elements.

    params.element_size    = af.sum((params.x_nodes[1] - params.x_nodes[0])\
                                                        / params.N_Elements)
    params.elements_xi_LGL = af.constant(0, params.N_Elements, params.N_LGL)
    params.elements        = utils.linspace(af.sum(params.x_nodes[0]),
                             af.sum(params.x_nodes[1] - params.element_size),\
                                                            params.N_Elements)
    params.np_element_array   = np.concatenate((af.transpose(params.elements),
                                   af.transpose(params.elements +\
                                                       params.element_size)))

    params.element_mesh_nodes = utils.linspace(af.sum(params.x_nodes[0]),
                                        af.sum(params.x_nodes[1]),\
                                               params.N_Elements + 1)

    params.element_array = af.transpose(af.np_to_af_array\
                                       (params.np_element_array))
    params.element_LGL   = wave_equation.mapping_xi_to_x(af.transpose\
                                          (params.element_array), params.xi_LGL)

    # The minimum distance between 2 mapped LGL points.
    params.delta_x = af.min((params.element_LGL - af.shift(params.element_LGL, 1, 0))[1:, :])

    # dx_dxi for elements of equal size.
    params. dx_dxi = af.mean(wave_equation.dx_dxi_numerical((params.element_mesh_nodes[0 : 2]),\
                                   params.xi_LGL))


    # The value of time-step.
    params.delta_t = params.delta_x / (4 * params.c)

    # Array of timesteps seperated by delta_t.
    params.time    = utils.linspace(0, int(params.total_time / params.delta_t) * params.delta_t,
                                                        int(params.total_time / params.delta_t))

    # Initializing the amplitudes. Change u_init to required initial conditions.
    if (wave=='sin'):
        params.u_init     = af.sin(2 * np.pi * params.element_LGL)

    if (wave=='gaussian'):
        params.u_init = np.e ** (-(params.element_LGL) ** 2 / 0.4 ** 2)

    params.u          = af.constant(0, params.N_LGL, params.N_Elements, params.time.shape[0],\
                                     dtype = af.Dtype.f64)
    params.u[:, :, 0] = params.u_init

    return



def test_LGL_points():
    '''
    Comparing the LGL nodes obtained by LGL_points with
    the reference nodes for N = 6
    '''
    reference_nodes  = \
        af.np_to_af_array(np.array([-1.,                 -0.7650553239294647,\
                                    -0.28523151648064504, 0.28523151648064504,\
                                     0.7650553239294647,  1. \
                                   ] \
                                  ) \
                         )

    calculated_nodes = (lagrange.LGL_points(6))
    assert(af.max(af.abs(reference_nodes - calculated_nodes)) <= 1e-14)


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
    threshold = 1e-9
    change_parameters(8, 10, 11, 'gaussian')
    nodes = np.array([7, 10], dtype = np.float64)
    test_nodes = af.interop.np_to_af_array(nodes)
    analytical_dx_dxi = 1.5
        
    check_dx_dxi = abs((af.statistics.mean(wave_equation.dx_dxi_numerical
                    (test_nodes,params.xi_LGL)) - analytical_dx_dxi)) <= threshold
    
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
    threshold = 6e-10

    change_parameters(8, 10, 11, 'gaussian')
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
    
    assert af.sum(af.abs(basis_array_analytical - params.lagrange_coeffs))\
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
    change_parameters(8, 10, 11, 'gaussian')
    
    referenceFluxIntegral = af.transpose(af.interop.np_to_af_array(np.array
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
    
    numerical_flux = wave_equation.volume_integral_flux(params.u[:, :, 0])
    assert (af.mean(af.abs(numerical_flux - referenceFluxIntegral)) < threshold)

def test_lax_friedrichs_flux():
    '''
    A test function to test the lax_friedrichs_flux function in wave_equation
    module.
    '''
    threshold = 1e-14
    
    params.c = 1
    
    f_i = wave_equation.lax_friedrichs_flux(params.u[:, :, 0])
    analytical_lax_friedrichs_flux = params.u[-1, :, 0]
    assert af.max(af.abs(analytical_lax_friedrichs_flux - f_i)) < threshold


def test_surface_term():
    '''
    A test function to test the surface_term function in the wave_equation
    module using analytical Lax-Friedrichs flux.
    '''
    threshold = 1e-13
    params.c = 1
    
    change_parameters(8, 10, 8, 'gaussian')
    
    analytical_f_i        = (params.u[-1, :, 0])
    analytical_f_i_minus1 = (af.shift(params.u[-1, :, 0], 0, 1))
    
    L_p_1                 = af.constant(0, params.N_LGL, dtype = af.Dtype.f64)
    L_p_1[params.N_LGL - 1] = 1
    
    L_p_minus1    = af.constant(0, params.N_LGL, dtype = af.Dtype.f64)
    L_p_minus1[0] = 1
    
    analytical_surface_term = af.blas.matmul(L_p_1, analytical_f_i)\
        - af.blas.matmul(L_p_minus1, analytical_f_i_minus1)
    
    numerical_surface_term = (wave_equation.surface_term(params.u[:, :, 0]))
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
    
    change_parameters(8, 10, 8, 'gaussian')

    u_n_A_matrix         = af.blas.matmul(wave_equation.A_matrix(),\
                                                  params.u[:, :, 0])
    volume_integral_flux = wave_equation.volume_integral_flux(params.u[:, :, 0])
    surface_term         = test_surface_term()
    b_vector_analytical  = u_n_A_matrix + (volume_integral_flux -\
                                    (surface_term)) * params.delta_t
    b_vector_array       = wave_equation.b_vector(params.u[:, :, 0])
    
    assert (b_vector_analytical - b_vector_array) < threshold

def test_integrate():
    '''
    Testing the integrate() function by passing coefficients
    of a polynomial and comparing it to the analytical result.
    '''
    threshold = 1e-14

    test_coeffs = af.np_to_af_array(np.array([7., 6, 4, 2, 1, 3, 9, 2]))
    # The coefficients of a test polynomial
    # `7x^7 + 6x^6 + 4x^5 + 2x^4 + x^3 + 3x^2 + 9x + 2`

    # Using integrate() function.

    calculated_integral = lagrange.integrate(af.transpose(test_coeffs))

    analytical_integral = 8.514285714285714

    assert (calculated_integral - analytical_integral) <= threshold


def test_interpolation():
    '''
    '''
    threshold = 8e-8

    xi_i  = af.flat(af.transpose(af.tile(params.xi_LGL, 1, params.N_LGL)))
    eta_j = af.tile(params.xi_LGL, params.N_LGL)
    f_ij  = np.e ** (xi_i + eta_j)
    interpolated_f = wave_equation_2d.lag_interpolation_2d(f_ij)
    xi  = utils.linspace(-1, 1, 8)
    eta = utils.linspace(-1, 1, 8)
    assert (af.mean(utils.polyval_2d(interpolated_f, xi, eta) - np.e**(xi+eta)) < threshold)


def test_Li_Lj_coeffs():
    '''
    '''
    threshold = 2e-9

    numerical_L3_xi_L4_eta_coeffs = wave_equation_2d.Li_Lj_coeffs()[:, :, 28]

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
