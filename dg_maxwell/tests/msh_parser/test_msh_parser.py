#! /usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
sys.path.insert(0, os.path.abspath('./'))

import arrayfire as af
import numpy as np

from dg_maxwell import msh_parser
from dg_maxwell import params

af.set_backend(params.backend)
af.set_device(params.device)

def test_read_order_2_msh():
    '''
    This test reads the mesh
    :download:`rectangular.msh <../dg_maxwell/tests/msh_parser/mesh/rectangular.msh>`
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


def test_edge_location():
    '''
    Tests the :py:meth:`dg_maxwell.msh_parser.edge_location` by testing against
    known valid and invalid test cases.
    '''
    left_edge = np.array([0, 1, 2])
    bottom_edge = np.array([2, 3, 4])
    right_edge = np.array([4, 5, 6])
    top_edge = np.array([0, 6, 7])

    invalid_edges = [[0], [1, 2, 3], [3, 4, 5], [5, 6, 7]]

    left_edge_test   = (msh_parser.edge_location(left_edge) == 0)
    bottom_edge_test = (msh_parser.edge_location(bottom_edge) == 1)
    right_edge_test  = (msh_parser.edge_location(right_edge) == 2)
    top_edge_test    = (msh_parser.edge_location(top_edge) == 3)

    invalid_edge_tests = []

    for invalid_edge in invalid_edges:
        invalid_edge_tests.append(msh_parser.edge_location(invalid_edge) == None)

    invalid_edge_tests = np.array(invalid_edge_tests)
    
    assert left_edge_test and bottom_edge_test and right_edge_test \
           and top_edge_test and np.all(invalid_edge_tests)


def test_interelement_relations():
    '''
    Tests the :py:meth:`dg_maxwell.msh_parser.interelement_relations` function
    by comparing its output to that of a reference inter-element relations array
    obtained by visual examinations for a :math:`3 \\times 3` element mesh.
    You may download the mesh file from this
    :download:`link <../dg_maxwell/tests/msh_parser/mesh/square_3_3.msh>`.
    '''
    
    ref_interelement_relations = np.array([[-1.,  3.,  1., -1.],
                                           [ 0.,  4.,  2., -1.],
                                           [ 1.,  5., -1., -1.],
                                           [-1.,  6.,  4.,  0.],
                                           [ 3.,  7.,  5.,  1.],
                                           [ 4.,  8., -1.,  2.],
                                           [-1., -1.,  7.,  3.],
                                           [ 6., -1.,  8.,  4.],
                                           [ 7., -1., -1.,  5.]])
    
    nodes, elements = msh_parser.read_order_2_msh('./dg_maxwell/tests/msh_parser/mesh/square_3_3.msh')
    interelement_relations = msh_parser.interelement_relations(elements)
    
    assert np.all(interelement_relations == ref_interelement_relations)