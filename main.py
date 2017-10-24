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
    print(af.mean(wave_equation.time_evolution()))
