import arrayfire as af
import numpy as np
import os
import numpy as np
import arrayfire as af
import matplotlib

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
from dg_maxwell import global_variables

af.set_backend(params.backend)


#print(af.mean(af.abs(advection_2d.u_analytical(0) - params.u_e_ij)))

gv = global_variables.advection_variables(params.N_LGL, params.N_quad,\
                                          params.x_nodes, params.N_Elements,\
                                          params.c, params.total_time, params.wave,\
                                          params.c_x, params.c_y, params.courant,\
                                          params.mesh_file, params.total_time_2d)


print('start')
print(af.info())
#print(wave_equation_2d.dx_deta(gv.nodes[gv.elements[0]][:, 1], gv.xi_i, gv.eta_j))
#print(gv.nodes[gv.elements[1]][:, 1])
#print(af.np_to_af_array(gv.nodes[gv.elements[1]]))#[:, 0])
elements_nodes = (af.reorder(af.transpose(af.np_to_af_array(gv.nodes[gv.elements[:]])), 0, 2, 1))#[0, :, :])
#print(elements_nodes)



#dxi_dx_elements = (wave_equation_2d.trial_dxi_dx(elements_nodes[:, 0, :],\
#                   elements_nodes[:, 1, :], gv.xi_i, gv.eta_j))
#print(dxi_dx_elements.shape)
#print((wave_equation_2d.trial_dxi_dx(elements_nodes[:, 0, :],
#       elements_nodes[:, 1, :], gv.xi_i, gv.eta_j)).shape)
#print(wave_equation_2d.F_xi(gv.u_e_ij, gv).shape)
#print(wave_equation_2d.F_eta(gv.u_e_ij, gv).shape)
#print(gv.sqrt_g.shape)
#print(wave_equation_2d.volume_integral(gv.u_e_ij, gv).shape)
#print(wave_equation_2d.surface_term_vectorized(gv.u_e_ij, gv).shape)

#advection_2d.time_evolution(gv)
#wave_equation_2d.time_evolution(gv)

#L1_norm = np.zeros([5])
#for LGL in range(3, 8):
#    print(LGL)
#    change_parameters(LGL)
#    L1_norm[LGL - 3] = (advection_2d.time_evolution())
#    print(L1_norm[LGL - 3])
#
#print(L1_norm)
#
#L1_norm = np.array([8.20284941e-02, 1.05582246e-02, 9.12125969e-04, 1.26001632e-04, 8.97007162e-06, 1.0576058855881385e-06])
#LGL = (np.arange(6) + 3).astype(float)
#normalization = 8.20284941e-02 / (3 ** (-3 * 0.85))
#pl.loglog(LGL, L1_norm, marker='o', label='L1 norm')
#pl.loglog(LGL, normalization * LGL ** (-0.85 * LGL), color='black', linestyle='--', label='$N_{LGL}^{-0.85 N_{LGL}}$')
#pl.title('L1 norm v/s $N_{LGL}$')
#pl.legend(loc='best')
#pl.xlabel('$N_{LGL}$ points')
#pl.ylabel('L1 norm of error')
#pl.show()
