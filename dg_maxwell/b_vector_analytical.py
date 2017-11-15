#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import arrayfire as af
import numpy as np
import os
from matplotlib import pyplot as pl

from dg_maxwell import wave_equation
from dg_maxwell import msh_parser
from dg_maxwell import wave_equation_2d
from dg_maxwell import isoparam
from dg_maxwell import lagrange
from dg_maxwell import params
from dg_maxwell import utils

def lag_polynomial(x):
    return
