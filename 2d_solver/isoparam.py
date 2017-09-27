#! /usr/bin/env python3
# -*- coding: utf-8 -*-

def isoparam_x(x_nodes, xi, eta):
    '''
    '''
    N_0 = (-1.0 / 4.0) * (1 - xi)  * (1 + eta) * (1 + xi - eta)
    N_1 = (1.0 / 2.0)  * (1 - xi)  * (1 - eta**2)
    N_2 = (-1.0 / 4.0) * (1 - xi)  * (1 - eta) * (1 + xi + eta)
    N_3 = (1.0 / 2.0)  * (1 - eta) * (1 - xi**2)
    N_4 = (-1.0 / 4.0) * (1 + xi)  * (1 - eta) * (1 - xi + eta)
    N_5 = (1.0 / 2.0)  * (1 + xi)  * (1 - eta**2)
    N_6 = (-1.0 / 4.0) * (1 + xi)  * (1 + eta) * (1 - xi - eta)
    N_7 = (1.0 / 2.0)  * (1 + eta) * (1 - xi**2)
    
    x = N_0 * x_nodes[0] \
      + N_1 * x_nodes[1] \
      + N_2 * x_nodes[2] \
      + N_3 * x_nodes[3] \
      + N_4 * x_nodes[4] \
      + N_5 * x_nodes[5] \
      + N_6 * x_nodes[6] \
      + N_7 * x_nodes[7]
    
    return x


def isoparam_y(y_nodes, xi, eta):
    '''
    '''
    return isoparam_x(y_nodes, xi, eta)
