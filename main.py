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





















#A_matrix  = wave_equation.A_matrix()
#A_inverse = af.inverse(A_matrix)
#b_vector  = wave_equation.b_vector(params.u_init)
#print(params.delta_t * af.matmul(A_inverse, b_vector))
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
u_e_ij = np.e ** (-(x_e_ij ** 2 + y_e_ij ** 2)/(0.4 ** 2))

#volume_integral = advection_2d.volume_integral(u_e_ij)
#surface_term    = advection_2d.surface_term(u_e_ij)
A_matrix        = advection_2d.A_matrix()
A_inverse       = af.inverse(A_matrix)
delta_t         = 1e-4
#
u_n_plus1 = u_e_ij + advection_2d.RK4_timestepping(A_inverse, u_e_ij, delta_t)
print(af.max(u_n_plus1))
print(af.min(u_n_plus1))
print(params.delta_t)
#
#U_1d_equi = (u_n_plus1[params.N_LGL - 1:params.N_LGL ** 2:params.N_LGL, :10])
#u_1d = params.u_init + wave_equation.RK4_timestepping(af.inverse(wave_equation.A_matrix()), params.u_init, 1e-4)
#print(u_1d)
#print(U_1d_equi - u_1d)
#
#pl.plot(params.element_LGL, af.abs(u_1d - U_1d_equi))
#pl.show()
#
#
color_levels = np.linspace(0., 1., 100)

#for i in range(100):
#    x_tile = af.moddims(x_e_ij[:, i], 8, 8)
#    y_tile = af.moddims(y_e_ij[:, i], 8, 8)
#    u_tile = af.moddims(u_e_ij[:, i], 8, 8)
#
#    x_tile = np.array(x_tile)
#    y_tile = np.array(y_tile)
#    u_tile = np.array(u_tile)
#    pl.contourf(x_tile, y_tile, u_tile, 200, levels = color_levels, cmap = 'jet')
#pl.show()
#
for i in range(100):
    x_tile = af.moddims(x_e_ij[:, i], 8, 8)
    y_tile = af.moddims(y_e_ij[:, i], 8, 8)
    u_n_plus1_tile = af.moddims(u_n_plus1[:, i], 8, 8)

    x_tile = np.array(x_tile)
    y_tile = np.array(y_tile)
    u_n_plus1_tile = np.array(u_n_plus1_tile)
    pl.contourf(x_tile, y_tile, u_n_plus1_tile, 200, levels = color_levels, cmap = 'jet')
pl.gca().set_aspect('equal')
pl.show()
