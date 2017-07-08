#! /usr/bin/env python3

import arrayfire as af

from app import lagrange
from utils import utils
from app import global_variables as gvar

def Li_Lp_xi(L_xi_j, L_xi_p):
	'''
	'''
	Li_Lp_xi = af.bcast.broadcast(utils.multiply, L_xi_j, L_xi_p)
	
	return Li_Lp_xi


def mappingXiToX(x_nodes, xi):
	'''
	'''
	N_0 = (1. - xi) / 2
	N_1 = (xi + 1.) / 2
	
	N0_x0 = af.bcast.broadcast(utils.multiply, N_0, x_nodes[0])
	N1_x1 = af.bcast.broadcast(utils.multiply, N_1, x_nodes[1])
	
	return N0_x0 + N1_x1


def dx_dxi(x_nodes, xi):
	'''
	'''
	dxi = 1e-8
	x2 = mappingXiToX(x_nodes, xi + dxi)
	x1 = mappingXiToX(x_nodes, xi - dxi)
	
	return (x2 - x1) / (2 * dxi)
