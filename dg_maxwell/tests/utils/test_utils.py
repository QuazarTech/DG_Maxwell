#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import arrayfire as af

from dg_maxwell import params
from dg_maxwell import utils
from dg_maxwell import lagrange

af.set_backend('opencl')
af.set_device(1)

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
