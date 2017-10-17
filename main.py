#! /usr/bin/env python3

import arrayfire as af
import numpy as np

from dg_maxwell import wave_equation
from dg_maxwell import lagrange
from dg_maxwell import isoparam
from dg_maxwell import params


if __name__ == '__main__':
    #print(af.mean(wave_equation.time_evolution()))
    f_coeffs = (af.np_to_af_array(np.array([[1, 2, 3, 4.], [5, 6, 7, 8]])))
    g_coeffs = (af.np_to_af_array(np.array([[1, 2, 3., 4], [5, 6, 7, 8]])))
    print(f_coeffs, g_coeffs)
    print(lagrange.integrate_2D(f_coeffs, g_coeffs))
