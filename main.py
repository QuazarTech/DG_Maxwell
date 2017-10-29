import arrayfire as af
import numpy as np
import os
from matplotlib import pyplot as pl

from dg_maxwell import params
from dg_maxwell import lagrange
from dg_maxwell import wave_equation_2d
from dg_maxwell import wave_equation
from dg_maxwell import isoparam
from dg_maxwell import msh_parser

nodes, elements = msh_parser.read_order_2_msh('Square_N_Elements_9.msh')
xi = af.flat(af.transpose(af.tile(params.xi_LGL, 1, params.N_LGL)))
eta = af.tile(params.xi_LGL, params.N_LGL)

L_p_1 = lagrange.lagrange_function_value(params.lagrange_coeffs)[:, 0]
L_p_1 = af.flat(af.transpose(af.tile(L_p_1, 1, params.N_LGL)))

L_p_minus1 = lagrange.lagrange_function_value(params.lagrange_coeffs)[:, -1]
L_p_minus1 = af.flat(af.transpose(af.tile(L_p_minus1, 1, params.N_LGL)))

L_q_1 = lagrange.lagrange_function_value(params.lagrange_coeffs)[:, 0]
L_q_1 = af.tile(L_q_1, params.N_LGL)

L_q_minus_1 = lagrange.lagrange_function_value(params.lagrange_coeffs)[:, -1]
L_q_minus_1 = af.tile(L_q_minus_1, params.N_LGL)

mapped_x_coords = isoparam.isoparam_x_2D(nodes[elements[0]][:, 0], xi, eta)
mapped_y_coords = isoparam.isoparam_x_2D(nodes[elements[0]][:, 1], xi, eta)
print(mapped_x_coords)

