#! /usr/bin/env python3

import arrayfire as af
af.set_backend('opencl')
af.set_device(1)

from app import global_variables as gvar
from app import wave_equation
from unit_test import test_waveEqn

if __name__ == '__main__':

    gvar.populateGlobalVariables(8)
    wave_equation.time_evolution()
