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
    
    wave_equation.numerical_A_matrix()
    #print(gvar.lagrange_value_array)
    print(test_waveEqn.test_isoparam_x())
