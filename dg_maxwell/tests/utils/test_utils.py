#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import arrayfire as af

from dg_maxwell import params
from dg_maxwell import utils
from dg_maxwell import lagrange

af.set_backend(params.backend)
af.set_device(params.device)

def test_matmul_3D():
    '''
    '''
    M = 3
    N = 2
    P = 4
    Q = 2

    a = af.range(M * N * Q, dtype = af.Dtype.u32)
    b = af.range(N * P * Q, dtype = af.Dtype.u32)

    a = af.moddims(a, d0 = M, d1 = N, d2 = Q)
    b = af.moddims(b, d0 = N, d1 = P, d2 = Q)

    a_init = a
    b_init = b
    
    ref_a_0 = np.matmul(np.array(a_init[:, :, 0]),
                        np.array(b_init[:, :, 0]))

    ref_a_1 = np.matmul(np.array(a_init[:, :, 1]),
                        np.array(b_init[:, :, 1]))
    
    test_matmul = np.array(utils.matmul_3D(a, b))
    
    diff_mat_0 = np.abs(test_matmul[:, :, 0] - ref_a_0)
    diff_mat_1 = np.abs(test_matmul[:, :, 1] - ref_a_1)
    
    assert np.all(diff_mat_0 == 0) and np.all(diff_mat_1 == 0)



def test_poly1d_prod():
    '''
    Checks the product of the polynomials of different degrees using the
    poly1d_product function and compares it to the analytically calculated
    product coefficients.
    '''
    
    N      = 3

    N_a    = 3
    poly_a = af.range(N * N_a, dtype = af.Dtype.u32)
    poly_a = af.moddims(poly_a, d0 = N, d1 = N_a)

    N_b    = 2
    poly_b = af.range(N * N_b, dtype = af.Dtype.u32)
    poly_b = af.moddims(poly_b, d0 = N, d1 = N_b)

    ref_poly = af.np_to_af_array(np.array([[0., 0., 9., 18.],
                                           [1., 8., 23., 28.],
                                           [4., 20., 41., 40.]]))

    test_poly1d_prod = utils.poly1d_product(poly_a, poly_b)
    test_poly1d_prod_commutative = utils.poly1d_product(poly_b, poly_a)

    diff     = af.abs(test_poly1d_prod - ref_poly)
    diff_commutative = af.abs(test_poly1d_prod_commutative - ref_poly)
    
    assert af.all_true(diff == 0.) and af.all_true(diff_commutative == 0.)



def test_integrate_1d():
    '''
    Tests the ``integrate_1d`` by comparing the integral agains the
    analytically calculated integral. The polynomials to be integrated
    are all the Lagrange polynomials obtained for the LGL points.
    
    The analytical integral is calculated in this `sage worksheet`_
    
    .. _sage worksheet: https://goo.gl/1uYyNJ
    '''
    
    threshold = 1e-12
    
    N_LGL     = 8
    xi_LGL    = lagrange.LGL_points(N_LGL)
    eta_LGL   = lagrange.LGL_points(N_LGL)
    _, Li_xi  = lagrange.lagrange_polynomials(xi_LGL)
    _, Lj_eta = lagrange.lagrange_polynomials(eta_LGL)

    Li_xi  = af.np_to_af_array(Li_xi)
    Lp_xi  = Li_xi.copy()
    
    Li_Lp = utils.poly1d_product(Li_xi, Lp_xi)
    
    test_integral_gauss = utils.integrate_1d(Li_Lp, order = 9,
                                             scheme = 'gauss')
    
    test_integral_lobatto = utils.integrate_1d(Li_Lp, order = N_LGL + 1,
                                               scheme = 'lobatto')

    
    ref_integral = af.np_to_af_array(np.array([0.0333333333333,
                                               0.196657278667,
                                               0.318381179651,
                                               0.384961541681,
                                               0.384961541681,
                                               0.318381179651,
                                               0.196657278667,
                                               0.0333333333333]))
    
    diff_gauss   = af.abs(ref_integral - test_integral_gauss)
    diff_lobatto = af.abs(ref_integral - test_integral_lobatto)
    
    assert af.all_true(diff_gauss < threshold) and af.all_true(diff_lobatto < threshold)

def test_polynomial_product_coeffs():
    '''
    '''
    threshold = 1e-12

    poly1 = af.reorder(
        af.transpose(
            af.np_to_af_array(np.array([[1, 2, 3., 4],
                                        [5, -2, -4.7211, 2]]))),
            0, 2, 1)
    
    poly2 = af.reorder(
        af.transpose(
            af.np_to_af_array(np.array([[-2, 4, 7., 9],
                                        [1, 0, -9.1124, 7]]))),
            0, 2, 1)

    numerical_product_coeffs    = utils.polynomial_product_coeffs(poly1, poly2)
    analytical_product_coeffs_1 = af.np_to_af_array(
        np.array([[-2, -4, -6, -8],
                  [4, 8, 12, 16],
                  [7, 14, 21, 28],
                  [9, 18, 27, 36]]))

    analytical_product_coeffs_2 = af.np_to_af_array(
        np.array([[5, -2, -4.7211, 2],
                  [0, 0, 0, 0],
                  [-45.562, 18.2248, 43.02055164, -18.2248],
                  [35, -14, -33.0477, 14]]))
    
    print(numerical_product_coeffs)
    assert af.max(af.abs(numerical_product_coeffs[:, :, 0] - analytical_product_coeffs_1 + \
                  numerical_product_coeffs[:, :, 1] - analytical_product_coeffs_2)) < threshold

def test_polyval_2d():
    '''
    Tests the ``utils.polyval_2d`` function by evaluating the polynomial
    
    .. math:: P_0(\\xi) P_1(\\eta)
    
    here,
    
    .. math:: P_0(\\xi) = 3 \, \\xi^{2} + 2 \, \\xi + 1
    
    .. math:: P_1(\\eta) = 3 \, \\eta^{2} + 2 \, \\eta + 1
    
    at corresponding ``linspace`` points in :math:`\\xi \\in [-1, 1]` and
    :math:`\\eta \\in [-1, 1]`.
    
    This value is then compared with the reference value calculated analytically.
    The reference values are calculated in
    `polynomial_product_two_variables.sagews`_
    
    .. _polynomial_product_two_variables.sagews: https://goo.gl/KwG7k9
    
    '''
    threshold = 1e-12
    
    poly_xi_degree = 4
    poly_xi = af.flip(af.np_to_af_array(np.arange(1, poly_xi_degree)))

    poly_eta_degree = 4
    poly_eta = af.flip(af.np_to_af_array(np.arange(1, poly_eta_degree)))
    
    poly_xi_eta = utils.polynomial_product_coeffs(poly_xi, poly_eta)
    
    xi  = utils.linspace(-1, 1, 8)
    eta = utils.linspace(-1, 1, 8)
    
    polyval_xi_eta = af.transpose(utils.polyval_2d(poly_xi_eta, xi, eta))
    
    polyval_xi_eta_ref = af.np_to_af_array(
        np.array([4.00000000000000,
                  1.21449396084962,
                  0.481466055810080,
                  0.601416076634741,
                  1.81424406497291,
                  5.79925031236988,
                  15.6751353602663,
                  36.0000000000000]))
    
    diff = af.abs(polyval_xi_eta - polyval_xi_eta_ref)
    
    assert af.all_true(diff < threshold)
