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

#nodes, elements = msh_parser.read_order_2_msh('Square_N_Elements_9.msh')
#wave_equation.time_evolution()
print((wave_equation.volume_integral_flux(params.u_init)))
#print(lagrange.integrate(params.lagrange_coeffs))
#print(wave_equation_2d.A_matrix())
print(params.u_init)
pl.plot((params.element_LGL), (params.u_init))
pl.show()
