# DG_Maxwell
 ```
 _____   _____   __  __                          _ _ 
 |  __ \ / ____| |  \/  |                        | | |
 | |  | | |  __  | \  / | __ ___  ____      _____| | |
 | |  | | | |_ | | |\/| |/ _` \ \/ /\ \ /\ / / _ \ | |
 | |__| | |__| | | |  | | (_| |>  <  \ V  V /  __/ | |
 |_____/ \_____| |_|  |_|\__,_/_/\_\  \_/\_/ \___|_|_|
 ```

## Introduction
This project is a stepping stone to develop a solver for solving the 2D Maxwell's equations using the Discontinuous Galerkin method. The project to develop 2D wave equation solver is a stepping stone to develop the Maxwell's equation solver. The 2D wave equation is given by the equation:

![2d_wave_eqn](./.svgs/2d_wave_eqn.svg "2D Wave equation" )

where,

![u](./.svgs/u.svg "u" )

![F](./.svgs/F.svg "F" )

![c](./.svgs/c.svg "c" )

The code and the algorithms developed while creating the 2D wave equation solver will help in creating the
Maxwell's equation solver.

Currently the 2D wave equation solver will focus on solving the wave equation in a rectangular domain in the
![xy](./.svgs/x_y.svg "x_y") plane, with periodic boundary conditions.
