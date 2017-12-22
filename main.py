#! /usr/bin/env python3

import numpy as np
import arrayfire as af

af.set_backend('cpu')
af.set_device(0)

from dg_maxwell import params
from dg_maxwell import wave_equation
from dg_maxwell import lagrange

if __name__ == '__main__':
    # 1. Set the initial conditions
    
    ## 1. Sin
    #E_00 = 1.
    #E_01 = 1.

    #B_00 = 0.2
    #B_01 = 0.5

    #E_z_init = E_00 * af.sin(2 * np.pi * params.element_LGL) \
            #+ E_01 * af.cos(2 * np.pi * params.element_LGL)

    #B_y_init = B_00 * af.sin(2 * np.pi * params.element_LGL) \
            #+ B_01 * af.cos(2 * np.pi * params.element_LGL)
    
    ## 2. Gaussian
    E_0 = 1.
    B_0 = 1.
    
    sigma = 0.1
    
    E_z_init = E_0 * np.e**(-(params.element_LGL)**2 / sigma**2)
    B_y_init = B_0 * np.e**(-(params.element_LGL)**2 / sigma**2)
    
    
    ###############################################################
    ####################### SET u_init ############################
    ###############################################################
    u_init = af.constant(0., d0 = params.N_LGL, d1 = params.N_Elements, d2 = 2)
    u_init[:, :, 0] = E_z_init
    u_init[:, :, 1] = B_y_init
    
    #u_init = params.u_init
    
    u_diff = wave_equation.time_evolution(u_init)
