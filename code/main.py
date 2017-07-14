#! /usr/bin/env python3

import arrayfire as af
af.set_backend('opencl')

from app import global_variables as gvar
from unit_test import test_waveEqn

if __name__ == '__main__':
	'''
	Main Function which sets backend and pouplates the global global_variables.
	Can be used to obtain results from other modules.
	'''
	gvar.populateGlobalVariables(8)
	test_waveEqn.test_gaussQuadLiLp()
