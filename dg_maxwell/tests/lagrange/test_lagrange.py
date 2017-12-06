#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from scipy import special as sp
import arrayfire as af

from dg_maxwell import utils
from dg_maxwell import params
from dg_maxwell import lagrange

af.set_backend(params.backend)
af.set_device(params.device)

def test_Li_basis_value():
    '''
    This test compares the output of the lagrange basis value calculated by the
    function ``Li_basis_value`` to the analytical value of the lagrange
    polynomials created using the same LGL points.
    The analytical values were calculated in this sage worksheet_
    
    .. _worksheet: https://goo.gl/ADyA3U
    '''
    
    threshold = 1e-11
    
    Li_value_ref = af.np_to_af_array(utils.csv_to_numpy(
        'dg_maxwell/tests/lagrange/files/Li_value.csv'))
    
    N_LGL = 8
    xi_LGL = lagrange.LGL_points(N_LGL)
    L_basis_poly1d, L_basis_af = lagrange.lagrange_polynomials(xi_LGL)
    L_basis_af = af.np_to_af_array(L_basis_af)
    
    Li_indexes = af.np_to_af_array(np.arange(3, dtype = np.int32))
    xi = af.np_to_af_array(np.linspace(-1., 1, 10))
                           
    Li_value = lagrange.Li_basis_value(L_basis_af, Li_indexes, xi)
    
    assert af.all_true(af.abs(Li_value - Li_value_ref) < threshold)
