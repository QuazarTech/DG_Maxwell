#! /usr/bin/env python3

import arrayfire as af
af.set_backend('opencl')
import numpy as np
from app import global_variables as gvar
from unit_test import test_waveEqn
from app import wave_equation
from app import lagrange
if __name__ == '__main__':
	
	gvar.populateGlobalVariables(8)
	print(af.timeit(wave_equation.elementFluxIntegral,\
		af.range(gvar.N_Elements)))
	print(wave_equation.surface_term(0))
