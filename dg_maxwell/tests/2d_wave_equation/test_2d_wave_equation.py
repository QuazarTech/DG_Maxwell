#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
sys.path.insert(0, os.path.abspath('./'))

import arrayfire as af
af.set_backend('cpu')
import numpy as np

from dg_maxwell import isoparam

def test_dx_dxi():
    '''
    '''
    
    
    
    return