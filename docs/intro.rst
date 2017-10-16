============
Introduction
============
This projects aims at developing a fast Maxwell's Equation solver using
the discontinuous Galerkin method. As first steps to develop this solver,
we are developing a library which allows us to solve the 1D wave equation
and the 2D wave equation using discontinuous Galerkin method.

1D Wave Eqaution Solver
-----------------------
The 1D wave equation solver is aimed at finding the time evolution of
the 1D wave equation using the discontinuous Galerkin method.
The 1D wave equation is given by the equation:

.. math:: \frac{\partial u}{\partial t} + \frac{\partial F}{\partial x} = 0
    :label: 1d_wave_eq

where,

:math:`u \equiv u(x, t)`

:math:`F(u) = cu`

where :math:`c` is the wave speed.

2D Wave Equation Solver
-----------------------

The 2D wave equation solver is aimed at finding the time evolution of the
2D wave equation using the discontinuous Galerkin method.
The 2D wave equation is given by the equation:

.. math:: \frac{\partial u}{\partial t} + \vec{\nabla} \cdot \vec{F} = 0
    :label: 2d_wave_eq

where,

:math:`u \equiv u(x, y, t)`

:math:`\vec{F} = \vec{c}u`

:math:`\vec{c} = c_0\hat{i} + c_1\hat{j}`

where, :math:`c_0` and :math:`c_1` denotes
the component of the wave speed in the :math:`\hat{i}`
and :math:`\hat{j}` direction respectively.

Through the development of the 2D wave equation solver, the code and
the algorithms developed here will help in creating the 2D Maxwell's
equation solver. Currently the 2D wave equation solver will focus on
solving the wave equation in a rectangular domain in the
:math:`(x, y)` plane, with periodic boundary conditions.