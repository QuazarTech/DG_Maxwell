#! /usr/bin/env python3

import arrayfire as af
import numpy as np
af.set_backend('opencl')
from matplotlib import pyplot as plt

from app import params
from unit_test import test_waveEqn
from app import wave_equation
from app import lagrange
from utils import utils

# [TODO] - Set N_Elements = 12

if __name__ == '__main__':
    #print(wave_equation.time_evolution())
    wave_equation.convergence_test()
    #print(wave_equation.amplitude_quadrature_points(0))
