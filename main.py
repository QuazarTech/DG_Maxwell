import numpy as np
import arrayfire as af


from dg_maxwell import params
from dg_maxwell import lagrange
from dg_maxwell import wave_equation
from dg_maxwell import msh_parser
from dg_maxwell import wave_equation_2d
from dg_maxwell import isoparam
from dg_maxwell import utils

af.set_backend(params.backend)

xi_i   = af.flat(af.transpose(af.tile(params.xi_LGL, 1, params.N_LGL)))
eta_j  = af.tile(params.xi_LGL, params.N_LGL)

nodes, elements = msh_parser.read_order_2_msh('square_1.msh')

u_ij   = np.e ** (- (xi_i ** 2) / (0.6 ** 2))
print(wave_equation_2d.surface_term(u_ij, nodes, elements))
test = af.range(9)
print(test, test[0:9:2])
