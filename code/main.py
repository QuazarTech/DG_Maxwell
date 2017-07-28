#! /usr/bin/env python3

import arrayfire as af
af.set_backend('cuda')

import numpy as np
from app import global_variables as gvar
from unit_test import test_waveEqn
from app import wave_equation
from app import lagrange


if __name__ == '__main__':
	gvar.populateGlobalVariables(8)
	element1_x_nodes = af.reorder(gvar.element_nodes[0 : 1], 1, 0, 2)
	print(wave_equation.volume_integral_flux(element1_x_nodes, gvar.u[0, :, 0]))
