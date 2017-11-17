import arrayfire as af
import numpy as np
import os
import numpy as np
import arrayfire as af

from matplotlib import pyplot as pl

from dg_maxwell import wave_equation
from dg_maxwell import msh_parser
from dg_maxwell import wave_equation_2d
from dg_maxwell.tests import test_waveEqn
from dg_maxwell import isoparam
from dg_maxwell import lagrange
from dg_maxwell import params
from dg_maxwell import advection_2d
from dg_maxwell import utils

af.set_backend(params.backend)

N_LGL = 8
#
nodes, elements = msh_parser.read_order_2_msh('square_10_10.msh')
#
x_e_ij  = af.np_to_af_array(np.zeros([N_LGL * N_LGL, len(elements)]))
y_e_ij  = af.np_to_af_array(np.zeros([N_LGL * N_LGL, len(elements)]))
#
xi_LGL  = lagrange.LGL_points(params.N_LGL)
eta_LGL = lagrange.LGL_points(params.N_LGL)
#
Xi_LGL, Eta_LGL = utils.af_meshgrid(xi_LGL, eta_LGL)
xi_i  = af.flat(Xi_LGL)
eta_j = af.flat(Eta_LGL)
#
for element_tag, element in enumerate(elements):
    x_e_ij[:, element_tag] = isoparam.isoparam_x_2D(nodes[element, 0], xi_i, eta_j)
    y_e_ij[:, element_tag] = isoparam.isoparam_y_2D(nodes[element, 1], xi_i, eta_j)
#
u_e_ij = np.e ** (-(x_e_ij ** 2 + y_e_ij ** 2)/(0.4 ** 2))

print(advection_2d.volume_integction_2d.volume_integral(u))
