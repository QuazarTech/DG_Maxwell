import numpy as np
import arrayfire as af
import math
from matplotlib import pyplot as plt
from app import lagrange
from app import global_variables as gvar
from app import wave_equation
from utils import utils
af.set_backend('opencl')
af.set_device(1)
