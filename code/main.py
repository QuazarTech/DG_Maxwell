#! /usr/bin/env python3

from os import sys

import arrayfire as af
af.set_backend('cuda')

from app import global_variables as gvar
from app import wave_equation
from unit_test import test_waveEqn

if __name__ == '__main__':
	
	gvar.populateGlobalVariables(9)
	test_waveEqn.test_timeEvolutionAnalyticSurfaceTerm()
	#wave_equation.time_evolution()
