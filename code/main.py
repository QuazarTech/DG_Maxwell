#! /usr/bin/env python3

import arrayfire as af
af.set_backend('opencl')
af.set_device(1)
import numpy as np
from app import global_variables as gvar
from unit_test import test_waveEqn
from app import wave_equation
from app import lagrange


if __name__ == '__main__':
	
	gvar.populateGlobalVariables(8)
	print(gvar.time.shape)
	wave_equation.time_evolution()
