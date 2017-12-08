#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import arrayfire as af
af.set_backend('opencl')
af.set_device(0)

from dg_maxwell import params
