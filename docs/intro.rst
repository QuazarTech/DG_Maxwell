Introduction
------------

This project is a stepping stone to develop a solver for solving
the 2D Maxwell's equations using the Discontinuous Galerkin method.
The project to develop 2D wave equation solver is a stepping stone
to develop the Maxwell's equation solver. The 2D wave equation is
given by the equation:

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