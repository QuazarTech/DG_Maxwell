#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
sys.path.insert(0, os.path.abspath('./2d_solver/'))

import numpy as np

import lagrange

def test_LGL_points():
    '''
    Comparing the LGL nodes obtained by LGL_points with
    the analytically found LGL points for :math:`N_{LGL} = 6`
    '''
    threshold = 1e-14
    
    reference_nodes  = np.array([-1., -0.7650553239294647,
                                 -0.28523151648064504, 0.28523151648064504,
                                 0.7650553239294647, 1.])
    
    N_LGL = 6
    calculated_nodes = lagrange.LGL_points(N_LGL)
    
    assert np.sum(np.abs(reference_nodes - calculated_nodes)) < threshold
