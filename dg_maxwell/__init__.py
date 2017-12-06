#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import arrayfire as af

from dg_maxwell import params

af.set_backend(params.backend)
af.set_device(params.device)
