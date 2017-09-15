#! /usr/bin/env python3
import math

import numpy as np
import arrayfire as af
from matplotlib import pyplot as plt
af.set_backend('opencl')

from app import params
from app import lagrange
from app import wave_equation
from utils import utils

# This test uses the initial paramters N_LGL = 8, N_Elements = 10

def test_mapping_xi_to_x():
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
    numerical_x_value  = wave_equation.mapping_xi_to_x(test_element_nodes, test_xi)
    
    assert af.abs(analytical_x_value - numerical_x_value) <= threshold


def test_dx_dxi():
    '''
    A Test function to check the dx_xi function in wave_equation module by
    passing nodes of an element and using the LGL points. Analytically, the
    differential would be a constant. The check has a tolerance 1e-7.
    '''
    threshold = 1e-14
    nodes = np.array([7, 10], dtype = np.float64)
    test_nodes = af.interop.np_to_af_array(nodes)
    analytical_dx_dxi = 1.5
    
    check_dx_dxi = (af.statistics.mean(wave_equation.dx_dxi_numerical
                    (test_nodes,params.xi_LGL)) - analytical_dx_dxi) <= threshold
    
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
    
    assert af.sum(af.abs(basis_array_analytical - params.lagrange_coeffs)) < threshold


def test_dLp_xi():
    '''
    Test function to check the dLp_xi calculated in params module with a
    numerically obtained one.

    Refrence
    --------
    The link to the sage worksheet where the calculations were carried out.
    `https://goo.gl/Xxp3PT`
    '''
    threshold = 4e-11
    
    reference_d_Lp_xi = af.np_to_af_array(np.array([\
    [-14.0000000000226,-3.20991570302344,0.792476681323880,-0.372150435728984,\
    0.243330712724289,-0.203284568901545,0.219957514771985, -0.500000000000000],
    
    [18.9375986071129, 3.31499272476776e-11, -2.80647579473469,1.07894468878725\
    ,-0.661157350899271,0.537039586158262, -0.573565414940005,1.29768738831567],

    [-7.56928981931106, 4.54358506455201, -6.49524878326702e-12, \
    -2.37818723350641, 1.13535801687865, -0.845022556506714, 0.869448098330221,\
    -1.94165942553537],

    [4.29790816425547,-2.11206121431525,2.87551740597844,-1.18896004153157e-11,\
    -2.38892435916370, 1.37278583181113, -1.29423205091574, 2.81018898925442],

    [-2.81018898925442, 1.29423205091574, -1.37278583181113, 2.38892435916370, \
    1.18892673484083e-11,-2.87551740597844, 2.11206121431525,-4.29790816425547],

    [1.94165942553537, -0.869448098330221, 0.845022556506714,-1.13535801687865,\
     2.37818723350641, 6.49524878326702e-12,-4.54358506455201,7.56928981931106],\

    [-1.29768738831567, 0.573565414940005,-0.537039586158262,0.661157350899271,\
    -1.07894468878725,2.80647579473469,-3.31498162253752e-11,-18.9375986071129],

    [0.500000000000000,-0.219957514771985,0.203284568901545,-0.243330712724289,\
    0.372150435728984, -0.792476681323880, 3.20991570302344, 14.0000000000226]
    ]))
        
    assert af.max(reference_d_Lp_xi - params.dLp_xi) < threshold


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
    threshold = 4e-8
    params.c = 1
    
    referenceFluxIntegral = af.transpose(af.interop.np_to_af_array(np.array([
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
        0.00236838757932, 0.00130167737191, 0.000588597708116, 0.00201663487667\
            ]])))
    
    numerical_flux = wave_equation.volume_integral_flux(params.u[:, :, 0])
    assert (af.max(af.abs(numerical_flux - referenceFluxIntegral)) < threshold)

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
    
    u_n_A_matrix         = af.blas.matmul(wave_equation.A_matrix(), params.u[:, :, 0])
    volume_integral_flux = wave_equation.volume_integral_flux(params.u[:, :, 0])
    surface_term         = test_surface_term()
    b_vector_analytical  = u_n_A_matrix + (volume_integral_flux -\
                                    (surface_term)) * params.delta_t
    b_vector_array       = wave_equation.b_vector(params.u[:, :, 0])
    
    assert (b_vector_analytical - b_vector_array) < threshold
