#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
sys.path.insert(0, os.path.abspath('./'))

import arrayfire as af
af.set_backend('opencl')
af.set_device(1)

import numpy as np

from dg_maxwell import msh_parser

def test_read_order_2_msh():
    '''
    This test reads the mesh
    :download:`rectangular.msh <./2d_solver/tests/msh_parser/mesh/rectangular.msh>`
    and stores them in the variables nodes and elements. The variables nodes
    and elements are compared against the test_nodes and test_elements
    respectively, which are created by manually reading the mesh file.
    '''
    
    threshold = 1e-14
    
    test_nodes = [[-2, -1                                    ],
                  [2, -1                                     ],
                  [2, 1                                      ],
                  [-2, 1                                     ],
                  [-5.500488953202876e-12, -1                ],
                  [-1.000000000002208, -1                    ],
                  [0.9999999999972498, -1                    ],
                  [2, -2.625677453238495e-12                 ],
                  [5.500488953202876e-12, 1                  ],
                  [1.000000000002208, 1                      ],
                  [-0.9999999999972498, 1                    ],
                  [-2, 2.625677453238495e-12                 ],
                  [0, 0                                      ],
                  [0.9999999999997291, -1.312838726619248e-12],
                  [-0.9999999999997291, 1.312838726619248e-12]]
                  
    test_nodes = np.array(test_nodes)
    
    test_elements = [[2, 9, 8, 12, 4, 6, 1, 7, 13],
                     [8, 10, 3, 11, 0, 5, 4, 12, 14]]
    
    test_elements = np.array(test_elements)
    
    nodes, elements = msh_parser.read_order_2_msh(
        os.path.abspath('./dg_maxwell/tests/msh_parser/mesh/rectangular.msh'))
    
    node_test = np.all((nodes - test_nodes) < threshold)
    element_test = np.all((elements == test_elements))
    
    assert node_test & element_test
