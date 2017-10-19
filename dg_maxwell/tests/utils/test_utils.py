#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import arrayfire as af

from dg_maxwell import params
from dg_maxwell import utils

af.set_backend(params.backend)

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
    
    diff_mat_0 = test_matmul[:, :, 0] - ref_a_0
    diff_mat_1 = test_matmul[:, :, 1] - ref_a_1
    
    print(diff_mat_0)
    print(diff_mat_1)
    
    assert np.all(diff_mat_0 == 0) and np.all(diff_mat_1 == 0)

