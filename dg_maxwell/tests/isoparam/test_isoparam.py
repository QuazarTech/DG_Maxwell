#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
sys.path.insert(0, os.path.abspath('./'))

import arrayfire as af
import numpy as np

from dg_maxwell import isoparam
from dg_maxwell import params

af.set_backend(params.backend)
af.set_device(params.device)

def test_isoparam_x():
    '''
    This test tests the function ``isoparam_x`` function. It uses a list of
    analytically calculated values at this sage worksheet `isoparam.sagews`_
    for an element at :math:`5` random :math:`(\\xi, \\eta)` coordinates
    and finds the :math:`L_1` norm of the :math:`x` coordinates got by the
    ``isoparam_x`` function.

    .. _isoparam.sagews: https://goo.gl/3EP3Pg
    
    '''
    threshold = 1e-7

    x_nodes = (np.array([0., 0.2, 0., 0.5, 1., 0.8, 1., 0.5]))
    y_nodes = (np.array([1., 0.5, 0., 0.2, 0., 0.5,  1., 0.8]))

    xi = np.array([-0.71565335, -0.6604077, -0.87006188, -0.59216134,
                   0.73777285])
    eta = np.array([-0.76986362, 0.62167345, -0.38380703, 0.85833585,
                    0.92388897])

    Xi, Eta = np.meshgrid(xi, eta)
    Xi  = af.np_to_af_array(Xi) 
    Eta = af.np_to_af_array(Eta) 

    x = isoparam.isoparam_x_2D(x_nodes, Xi, Eta)
    y = isoparam.isoparam_y_2D(y_nodes, Xi, Eta)

    test_x = af.np_to_af_array(np.array( \
        [[ 0.20047188, 0.22359428, 0.13584604, 0.25215798, 0.80878597],
        [ 0.22998716, 0.2508311 , 0.1717295 , 0.27658015, 0.77835843],
        [ 0.26421973, 0.28242104, 0.21334805, 0.3049056 , 0.7430678 ],
        [ 0.17985384, 0.20456788, 0.11077948, 0.23509776, 0.83004127],
        [ 0.16313183, 0.18913674, 0.09044955, 0.22126127, 0.84728013]]))

    test_y = af.np_to_af_array(np.array(
        [[ 0.19018229, 0.20188751, 0.15248238, 0.2150496 , 0.18523221],
        [ 0.75018125, 0.74072916, 0.78062435, 0.73010062, 0.7541785 ],
        [ 0.34554379, 0.3513793 , 0.32674892, 0.35794112, 0.34307598],
        [ 0.84542176, 0.83237139, 0.88745412, 0.81769672, 0.8509407 ],
        [ 0.87180243, 0.85775537, 0.9170449 , 0.84195996, 0.87774287]]))

    L1norm_x_test_x = np.abs(x - test_x).sum()
    L1norm_y_test_y = np.abs(y - test_y).sum()

    assert (L1norm_x_test_x < threshold) & (L1norm_y_test_y < threshold)

