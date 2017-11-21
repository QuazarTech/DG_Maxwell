import arrayfire as af
import numpy as np
import os
import numpy as np
import arrayfire as af

from matplotlib import pyplot as pl
from tqdm import trange

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

print(af.info())
N_LGL = params.N_LGL
#
nodes, elements = msh_parser.read_order_2_msh('square_10_10.msh')
#
x_e_ij  = af.np_to_af_array(np.zeros([N_LGL * N_LGL, len(elements)]))
y_e_ij  = af.np_to_af_array(np.zeros([N_LGL * N_LGL, len(elements)]))
##
xi_LGL  = lagrange.LGL_points(params.N_LGL)
eta_LGL = lagrange.LGL_points(params.N_LGL)
##
Xi_LGL, Eta_LGL = utils.af_meshgrid(xi_LGL, eta_LGL)
xi_i  = af.flat(Xi_LGL)
eta_j = af.flat(Eta_LGL)
##
for element_tag, element in enumerate(elements):
    x_e_ij[:, element_tag] = isoparam.isoparam_x_2D(nodes[element, 0], xi_i, eta_j)
    y_e_ij[:, element_tag] = isoparam.isoparam_y_2D(nodes[element, 1], xi_i, eta_j)

u_e_ij = np.e ** (-(x_e_ij ** 2 + y_e_ij ** 2)/(0.4 ** 2))
print('start')
print(af.info())
#for i in range(10):
#    print(advection_2d.volume_integral(u_e_ij).shape)
##print(advection_2d.volume_integral(u_e_ij).shape)
A_inverse = (np.linalg.inv(np.array(advection_2d.A_matrix())))
A_inverse = af.np_to_af_array(A_inverse)
u         = u_e_ij
delta_t   = 1e-3
print(delta_t)
##
#for i in trange(1):
##    u += advection_2d.RK4_timestepping(A_inverse, u, delta_t)
#
#    #Implementing second order time-stepping.
#    u_n_plus_half =  u + af.matmul(A_inverse, advection_2d.b_vector(u))\
#                          * delta_t / 2
#
#    u            +=  af.matmul(A_inverse, advection_2d.b_vector(u_n_plus_half))\
#                      * delta_t
#
#    L1_norm = af.mean(af.abs(u_e_ij - u))
#    if (L1_norm >= 100):
#        break
#
#
color_levels = np.linspace(-0.1, 1.1, 100)
##
#L1_norm = af.mean(af.abs(u_e_ij - u))
#print((L1_norm))

u_plot = af.flip(af.moddims(u, params.N_LGL, params.N_LGL, 10, 10), 0)
x_plot = af.flip(af.moddims(x_e_ij, params.N_LGL, params.N_LGL, 10, 10), 0)
y_plot = af.flip(af.moddims(y_e_ij, params.N_LGL, params.N_LGL, 10, 10), 0)

print(x_plot[:, :, 0, 0], y_plot[:, :, 0, 0])

x_contour = af.np_to_af_array(np.zeros([params.N_LGL * 10, params.N_LGL * 10]))
y_contour = af.np_to_af_array(np.zeros([params.N_LGL * 10, params.N_LGL * 10]))
u_contour = af.np_to_af_array(np.zeros([params.N_LGL * 10, params.N_LGL * 10]))

for i in range(100):
    p = int(i / 10)
    q = i - p * 10
    x_contour[p * params.N_LGL:params.N_LGL * (p + 1),\
              q * params.N_LGL:params.N_LGL * (q + 1)] = x_plot[:, :, q, p]

    y_contour[p * params.N_LGL:params.N_LGL * (p + 1),\
              q * params.N_LGL:params.N_LGL * (q + 1)] = y_plot[:, :, q, p]


    u_contour[p * params.N_LGL:params.N_LGL * (p + 1),\
              q * params.N_LGL:params.N_LGL * (q + 1)] = u_plot[:, :, q, p]

print(x_contour[:, 0], y_contour[0, :])
x_contour = np.array(x_contour)
y_contour = np.array(y_contour)
u_contour = np.array(u_contour)
pl.contourf(x_contour, y_contour, u_contour, 200, levels = color_levels, cmap = 'jet')
pl.gca().set_aspect('equal')
pl.colorbar()
pl.show()
#for i in range(100):
#    x_tile = af.moddims(x_e_ij[:, i], params.N_LGL, params.N_LGL)
#    y_tile = af.moddims(y_e_ij[:, i], params.N_LGL, params.N_LGL)
#    u_tile = af.moddims(u[:, i], params.N_LGL, params.N_LGL)
#
#    x_tile = np.array(x_tile)
#    y_tile = np.array(y_tile)
#    u_tile = np.array(u_tile)
#    pl.contourf(x_tile, y_tile, u_tile, 200, levels = color_levels, cmap = 'jet')
#pl.gca().set_aspect('equal')
#pl.show()
