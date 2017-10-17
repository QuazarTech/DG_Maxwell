#! /usr/bin/env python3

import arrayfire as af
import numpy as np

from dg_maxwell import wave_equation
from dg_maxwell import lagrange
from dg_maxwell import isoparam
from dg_maxwell import params


if __name__ == '__main__':
#    test_nodes = af.np_to_af_array(np.array([[-2, 2],[-2, 0],[-2, -2]\
#            , [0., -2], [2, -2], [2, 0],[2, 2], [0, 2]]))
#    print(test_nodes[:, 0])
#    x_LGL_tile = af.flat(af.transpose(af.tile(params.xi_LGL, 1, params.N_LGL)))
#    y_LGL_tile = af.tile(params.xi_LGL, params.N_LGL)
#    x_LGL = (isoparam.isoparam_x_2D(test_nodes[:, 0], x_LGL_tile, y_LGL_tile))
#    y_LGL = (isoparam.isoparam_y_2D(test_nodes[:, 1], x_LGL_tile, y_LGL_tile))
#    u_ij  = af.np_to_af_array(np.zeros([64, 2]))
#    u_ij[:,0] = x_LGL
#    u_ij[:,1] = y_LGL
#    print(u_ij)
#    print(af.mean(wave_equation.time_evolution()))
    print(wave_equation.A_matrix())
