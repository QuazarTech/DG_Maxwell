#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import arrayfire as af
af.set_backend('cuda')

# Number of Legendre-Gauss-Lobatto points
N = 8