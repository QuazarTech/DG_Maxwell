import numpy as np
import arrayfire as af

from matplotlib import pyplot as pl

from dg_maxwell import params
from dg_maxwell import lagrange
from dg_maxwell import wave_equation
from dg_maxwell import msh_parser
from dg_maxwell import wave_equation_2d
from dg_maxwell.tests import test_waveEqn
from dg_maxwell import isoparam
from dg_maxwell import utils

af.set_backend(params.backend)

#error_gauss   = af.np_to_af_array(np.zeros([15]))
#error_lobatto = af.np_to_af_array(np.zeros([15]))
#n_quad        = af.np_to_af_array(np.arange(14)) + 3
#for i in range(3, 17):
#    test_waveEqn.change_parameters(8, 10, i, 'gaussian')
#    analytical_integral = -0.002016634876668093
#    numerical_integral  = wave_equation.volume_integral_flux(params.u_init)[0, 0]
#    
#    error_lobatto[i - 3] = af.abs(numerical_integral - analytical_integral)
#
#    test_waveEqn.change_parameters(8, 10, i, 'gaussian', 'gauss_quadrature')
#    analytical_integral = -0.002016634876668093
#    numerical_integral  = wave_equation.volume_integral_flux(params.u_init)[0, 0]
#
#    error_gauss[i - 3] = af.abs(numerical_integral - analytical_integral)
#
#pl.semilogy(np.array(n_quad), np.array(error_gauss[:-1]), marker='o', label='Gauss quadrature')
#pl.semilogy(np.array(n_quad), np.array(error_lobatto[:-1]), marker='o', label='Lobatto quadrature')
#pl.title(r'Error in $\int \frac{d L_0(\xi)}{d\xi} F(\xi) d\xi$ ($N_{LGL}$ = 8, Number of elements = 10)')
#pl.xlabel('$N_{quad}$')
#pl.ylabel('Error')
#pl.legend(loc='best')
#pl.show()
#af.display(error_gauss[:-1], 14)
#print(n_quad)





#xi_i   = af.flat(af.transpose(af.tile(params.xi_LGL, 1, params.N_LGL)))
#eta_j  = af.tile(params.xi_LGL, params.N_LGL)
#
#nodes, elements = msh_parser.read_order_2_msh('square_1.msh')
#
#u_ij   = np.e ** (- (xi_i ** 2 ) / (0.6 ** 2))
#surface_term = (wave_equation_2d.surface_term(u_ij))
##(wave_equation_2d.lax_friedrichs_flux(u_ij))
#analytical_surface_term_xi_1_boundary = af.np_to_af_array(np.array(
#                                        [0.000518137698965616, 0.00316169869369534, 0.00555818937089650,\
#                                         0.00772576775028549, 0.00957686227446707, 0.0110358302160314,\
#                                         0.0120429724206691, 0.0125570655998897, 0.0125570656010149,\
#                                         0.0120429724194322, 0.0110358302175448, 0.00957686227250284,\
#                                         0.00772576775253881 , 0.00555818936935162 , 0.00316169869299086,\
#                                         0.000518137700074415 ]))
#analytical_surface_term_xi_minus1_boundary = af.np_to_af_array(np.array(
#                                             [0.000518137699431318, 0.00316169869653708, 0.00555818937589221,\
#                                              0.00772576775722942, 0.00957686228307477, 0.0110358302259504, \
#                                              0.0120429724314933, 0.0125570656111760, 0.0125570656123012, \
#                                              0.0120429724302564, 0.0110358302274638, 0.00957686228111053, \
#                                              0.00772576775948273, 0.00555818937434732, 0.00316169869583259, \
#                                              0.000518137700540118]))
#print(af.sum(af.abs(analytical_surface_term_xi_1_boundary - surface_term[-params.N_LGL:])))
#print(af.sum(af.abs(analytical_surface_term_xi_minus1_boundary - surface_term[-params.N_LGL:])))
#print(surface_term)
#print(wave_equation_2d.time_evolution())
#print(wave_equation_2d.b_vector(params.u_init_2d))
#print(wave_equation.b_vector(params.u_init))
