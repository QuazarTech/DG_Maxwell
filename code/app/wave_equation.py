#! /usr/bin/env python3

import arrayfire as af

#from app import lagrange
from utils import utils
#from app import global_variables as gvar
def Li_Lp_xi(L_xi_i, L_xi_p):
	'''
	LaTex strings      # / used instead of \
	-------------
	To calculate $L_{p}(/xi) L_{i}(/xi)$ for N Lagrange-Gaussian-Lobatto points 
	in a $/xi$ space [-1, 1] and indices i and p ranging from 0 to N - 1, We 
	need to start by creating a 1 X N x N matrix of $L_{p}(/xi)$.
	
	/begin{align}
	L_{p}(/xi) = /begin{pmatrix}
	L_{0}(/xi_{0}) & L_{1}(/xi_{0}) & /cdots & L_{N - 1}(/xi_{0}) //
	L_{0}(/xi_{1}) & L_{1}(/xi_{1}) & /cdots & L_{N - 1}(/xi_{1}) //
	/vdots  & /vdots  & /ddots & /vdots  //
	L_{0}(/xi_{N - 1}) & L_{1}(/xi_{N - 1}) & /cdots & L_{N - 1}(/xi_{N - 1}) 
	/end{pmatrix}_{1 X N X N }
	/end{align}
	
	Now, we need to get a $L_{i}(/xi)$ matrix which is the transpose of $L_{p}
	(/xi)$ and reordered as /newline(1, 0, 2), where (0, 1, 2) is the normal 
	ordering (x, y, z)
	
	/begin{align}
	L_{i}(/xi) = /begin{pmatrix}
	L_{0}(/xi_{0}) & L_{0}(/xi_{1}) & /cdots & L_{0}(/xi_{N - 1}) //
	L_{1}(/xi_{0}) & L_{1}(/xi_{1}) & /cdots & L_{1}(/xi_{N - 1}) //
	/vdots  & /vdots  & /ddots & /vdots  //
	L_{N - 1}(/xi_{0}) & L_{N - 1}(/xi_{1}) & /cdots & L_{N - 1}(/xi_{N - 1}) 
	/end{pmatrix}_{N X 1 X N}
	/end{align}
	
	Now, the $L_{i}(/xi)$ array is broadcasted across the dimension 0 N times.
	The result is an/newline N X N X N matrix with elements repeating N times 
	in dimension 0.
	/begin{tikzpicture}[every node/.style={anchor=north east,fill=white, 
	minimum width=1.4cm, minimum height=7mm}]
	/matrix (mA) [draw,matrix of math nodes]
	{
	L_{0}(/xi_{N - 1}) & L_{1}(/xi_{N - 1}) & ... & L_{N - 1}(/xi_{N - 1}) //
	L_{0}(/xi_{N - 1}) & L_{1}(/xi_{N - 1}) & ... & L_{N - 1}(/xi_{N - 1}) //
	. & . & ... & . //
	L_{0}(/xi_{N - 1}) & L_{1}(/xi_{N - 1}) & ... & L_{N - 1}(/xi_{N - 1}) //
	};
	/matrix (mB) [draw,matrix of math nodes] at ($(mA.south west)+(2,0.7)$)
	{
	L_{0}(/xi_{i}) & L_{1}(/xi_{i}) & ... & L_{N - 1}(/xi_{i}) //
	L_{0}(/xi_{i}) & L_{1}(/xi_{i}) & ... & L_{N - 1}(/xi_{i}) //
	. & . & ... & . //
	L_{0}(/xi_{i}) & L_{1}(/xi_{i}) & ... & L_{N - 1}(/xi_{i}) //
	};
	/matrix (mC) [draw,matrix of math nodes] at ($(mB.south west)+(2,0.7)$)
	{
	L_{0}(/xi_{0}) & L_{1}(/xi_{0}) & ... & L_{N - 1}(/xi_{0}) //
	L_{0}(/xi_{0}) & L_{1}(/xi_{0}) & ... & L_{N - 1}(/xi_{0}) //
	. & . & ... & . //
	L_{0}(/xi_{0}) & L_{1}(/xi_{0}) & ... & L_{N - 1}(/xi_{0}) //
	};

	/draw[dashed](mA.north east)--(mC.north east);
	/draw[dashed](mA.north west)--(mC.north west);
	/draw[dashed](mA.south east)--(mC.south east);
	/end{tikzpicture}

	Now doing the same for $L_{i}(/xi)$, i.e., broadcasting it in dimension 1 
	N times. Another /newline N X N X N array would be obtained which can be 
	multipled with the previous N X N X N array which would give the required
	$L_{p}(/xi) L_{i}(/xi)$ matrix




	/begin{tikzpicture}[every node/.style={anchor=north east,fill=white, 
	minimum width=1.4cm,minimum 		height=7mm}]
	/matrix (mA) [draw,matrix of math nodes]
	{
	L_{0}(/xi_{N -1}) & L_{0}(/xi_{N -1}) & ... & L_{0}(/xi_{N -1}) //
	L_{1}(/xi_{N -1}) & L_{1}(/xi_{N -1}) & ... & L_{1}(/xi_{N -1}) //
	. & . & ... & . //
	L_{N - 1}(/xi_{N -1}) & L_{N - 1}(/xi_{N -1}) & ... & L_{N - 1}(/xi_{N -1})
	//
	};

	/matrix (mB) [draw,matrix of math nodes] at ($(mA.south west)+(2,0.7)$)
	{
	L_{0}(/xi_{i}) & L_{0}(/xi_{i}) & ... & L_{0}(/xi_{i}) //
	L_{1}(/xi_{i}) & L_{1}(/xi_{i}) & ... & L_{1}(/xi_{i}) //
	. & . & ... & . //
	L_{N - 1}(/xi_{i}) & L_{N - 1}(/xi_{i}) & ... & L_{N - 1}(/xi_{i}) //
	};

	/matrix (mC) [draw,matrix of math nodes] at ($(mB.south west)+(2,0.7)$)
	{
	L_{0}(/xi_{0}) & L_{0}(/xi_{0}) & ... & L_{0}(/xi_{0}) //
	L_{1}(/xi_{0}) & L_{1}(/xi_{0}) & ... & L_{1}(/xi_{0}) //
	. & . & ... & . //
	L_{N - 1}(/xi_{0}) & L_{N - 1}(/xi_{0}) & ... & L_{N - 1}(/xi_{0}) //
	};

	/draw[dashed](mA.north east)--(mC.north east);
	/draw[dashed](mA.north west)--(mC.north west);
	/draw[dashed](mA.south east)--(mC.south east);
	/end{tikzpicture}


	/begin{tikzpicture}[every node/.style={anchor=north east,fill=white, 
	minimum width=1.4cm,minimum height=10mm}]
	/matrix (mA) [draw,matrix of math nodes]
	{
	L_{0}(/xi_{N - 1})L_{0}(/xi_{N - 1}) & L_{0}(/xi_{N - 1})L_{1}
	(/xi_{N - 1}) & ... & L_{0}(/xi_{N - 1})L_{N - 1}(/xi_{N - 1}) //
	L_{1}(/xi_{N - 1})L_{0}(/xi_{N - 1}) & L_{1}(/xi_{N - 1})L_{1}
	(/xi_{N - 1}) & ... & L_{1}(/xi_{N - 1})L_{N - 1}(/xi_{N - 1}) //
	. & . & ... & . //
	L_{N - 1}(/xi_{N - 1})L_{0}(/xi_{N - 1}) & L_{N - 1}
	(/xi_{N - 1})L_{1}(/xi_{N - 1}) & ... & L_{N - 1}(/xi_{N - 1})L_{N - 1}
	(/xi_{N - 1}) //
	};

	/matrix (mB) [draw,matrix of math nodes] at ($(mA.south west)+(1.5,0.7)$)
	{
	L_{0}(/xi_{i})L_{0}(/xi_{i}) & L_{0}(/xi_{i})L_{1}(/xi_{i}) & ... & L_{0}
	(/xi_{i})L_{N - 1}(/xi_{i}) //
	L_{1}(/xi_{i})L_{0}(/xi_{i}) & L_{1}(/xi_{i})L_{1}(/xi_{i}) & ... & L_{1}
	(/xi_{i})L_{N - 1}(/xi_{i}) //
	. & . & ... & . //
	L_{N - 1}(/xi_{i})L_{i}(/xi_{i}) & L_{N - 1}(/xi_{i})L_{1}(/xi_{i}) & ... 
	& L_{N - 1}(/xi_{i})L_{N - 1}(/xi_{i}) //
	};

	/matrix (mC) [draw,matrix of math nodes] at ($(mB.south west)+(1.5,0.7)$)
	{
	L_{0}(/xi_{0})L_{0}(/xi_{0}) & L_{0}(/xi_{0})L_{1}(/xi_{0}) & ... & L_{0}
	(/xi_{0})L_{N - 1}(/xi_{0}) //
	L_{1}(/xi_{0})L_{0}(/xi_{0}) & L_{1}(/xi_{0})L_{1}(/xi_{0}) & ... & L_{1}
	(/xi_{0})L_{N - 1}(/xi_{0}) //
	. & . & ... & . //
	L_{N - 1}(/xi_{0})L_{0}(/xi_{0}) & L_{N - 1}(/xi_{0})L_{1}(/xi_{0}) & 
	... & L_{N - 1}(/xi_{0})L_{N - 1}(/xi_{0}) //
	};
	/draw[dashed](mA.north east)--(mC.north east);
	/draw[dashed](mA.north west)--(mC.north west);
	/draw[dashed](mA.south east)--(mC.south east);
	/end{tikzpicture}
	-----------------
	Parameters
	----------
	
	L_xi_i : arrayfire.Array [N N 1 1]
	
	L_xi_i is an N x N x 1 x 1 matrix of lagrange basis functions of N LGL 
	points with indices from 0 to N-1
	
	L_xi_p : arrayfire.Array [1 N N 1]
	
	L_xi_p is a 1 x N x N x 1 matrix which is the transpose of L_xi_i reordered
	as (2, 0, 1, 3)
	
	Returns
	-------
	
	Li_Lp_xi : arrayfire.Array [N N N 1]
			   An N x N x N x 1 matrix with elements L_xi_i * L_xi_p, where i,
			   p, xi all range from 0 to N-1.
	
	'''
	
	Li_Lp_xi = af.bcast.broadcast(utils.multiply, L_xi_i, L_xi_p)
	
	return Li_Lp_xi


def mappingXiToX(x_nodes, xi):
	'''
	Parameters
	----------
	
	x_nodes : arrayfire.Array 
			  Contains the nodes of the elements
	
	xi		: float
			  Value of xi in domain (-1, 1) which returns the corresponding 
			  x value in the
	
	Returns
	-------
	X value in the element with given nodes and xi.
	'''
	N_0 = (1. - xi) / 2
	N_1 = (xi + 1.) / 2
	
	N0_x0 = af.bcast.broadcast(utils.multiply, N_0, x_nodes[0])
	N1_x1 = af.bcast.broadcast(utils.multiply, N_1, x_nodes[1])
	
	return N0_x0 + N1_x1


def dx_dxi(x_nodes, xi):
	'''
	Differential calculated by central differential method about xi using the
	mappingXiToX function.
	
	Parameters
	----------
	
	x_nodes : arrayfire.Array
			  Contains the nodes of elements
	
	xi		: float
			  Value of xi
	  
	Returns
	-------
	Numerical value of differential of X w.r.t the given xi 
	'''
	dxi = 1e-8
	x2 = mappingXiToX(x_nodes, xi + dxi)
	x1 = mappingXiToX(x_nodes, xi - dxi)
	
	return (x2 - x1) / (2 * dxi)
