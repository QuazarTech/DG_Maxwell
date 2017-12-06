#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

import arrayfire as af

backend = 'opencl'
device = 0

af.set_backend(backend)
af.set_device(device)

from dg_maxwell import lagrange
from dg_maxwell import isoparam
from dg_maxwell import utils
from dg_maxwell import msh_parser
from dg_maxwell import wave_equation
from dg_maxwell import wave_equation_2d

# The domain of the function.
x_nodes    = af.np_to_af_array(np.array([-1., 1.]))

# The number of LGL points into which an element is split.
N_LGL      = 6

# Number of elements the domain is to be divided into.
N_Elements = 9

# The scheme to be used for integration. Values are either
# 'gauss_quadrature' or 'lobatto_quadrature'
scheme     = 'gauss_quadrature'

# The scheme to integrate the volume integral flux
volume_integral_scheme = 'lobatto_quadrature'

# The number quadrature points to be used for integration.
N_quad     = 6

# Wave speed.
c          = 1

# The total time for which the wave is to be evolved by the simulation.
total_time = 2.01

# The c_lax to be used in the Lax-Friedrichs flux.
c_lax      = c

# The wave to be advected is either a sin or a Gaussian wave.
# This parameter can take values 'sin' or 'gaussian'.
wave = 'sin'

c_x = 1.


# The parameters below are for 2D advection
# -----------------------------------------


########################################################################
#######################2D Wave Equation#################################
########################################################################

c_x = 1.
c_y = 1.

courant = 0.1

mesh_file = 'examples/read_and_plot_mesh/mesh/square_10_10.msh'


total_time_2d = 2.0

volume_integrand_scheme_2d = 'Lobatto'
