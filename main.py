#! /usr/bin/env python3

import arrayfire as af
import numpy as np
from scipy import special as sp

from dg_maxwell import wave_equation
from dg_maxwell import lagrange
from dg_maxwell import isoparam
from dg_maxwell import params
from dg_maxwell import wave_equation_2d
from dg_maxwell import utils


if __name__ == '__main__':
    f_coeffs = (af.np_to_af_array(np.array([[1, 2, 3, 4.], [5, 6, 7, 8], [1, 2, 3, 4.]])))
    g_coeffs = (af.np_to_af_array(np.array([[1, 2, 3., 4], [5, 6, 7, 8], [1, 2, 3, 4.]])))
    print(f_coeffs, g_coeffs)
    Integral = (lagrange.integrate_2D(f_coeffs, g_coeffs))
    af.display(Integral[0], 14)
    #print(af.mean(wave_equation.time_evolution()))
    #print(wave_equation_2d.A_matrix().shape)
