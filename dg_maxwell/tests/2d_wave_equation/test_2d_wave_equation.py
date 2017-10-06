#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
sys.path.insert(0, os.path.abspath('./'))
import csv

import arrayfire as af
af.set_backend('cpu')
import numpy as np

from dg_maxwell import utils
from dg_maxwell import msh_parser
from dg_maxwell import lagrange
from dg_maxwell import wave_equation_2d

def test_dx_dxi():
    '''
    '''
    threshold = 1e-6
    
    dx_dxi_reference = utils.csv_to_numpy(
        'dg_maxwell/tests/2d_wave_equation/files/dx_dxi_data.csv')
    
    nodes, elements = msh_parser.read_order_2_msh(
        'dg_maxwell/tests/2d_wave_equation/files/circle.msh')
    
    
    N_LGL = 16
    xi_LGL  = np.array(lagrange.LGL_points(N_LGL))
    eta_LGL = np.array(lagrange.LGL_points(N_LGL))
    Xi, Eta = np.meshgrid(xi_LGL, eta_LGL)
    
    dx_dxi = wave_equation_2d.dx_dxi(
        nodes[elements[0]][:, 0], Xi, Eta)
    
    assert np.all(np.isclose(dx_dxi, dx_dxi_reference))
    
    