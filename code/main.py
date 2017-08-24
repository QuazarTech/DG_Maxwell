#! /usr/bin/env python3

from os import sys
import numpy as np
import arrayfire as af
af.set_backend('opencl')
from app import global_variables as gvar
from app import lagrange
from app import wave_equation
from unit_test import test_waveEqn



if __name__ == '__main__':
	
	gvar.populateGlobalVariables(3)
	print(wave_equation.numerical_A_matrix())
