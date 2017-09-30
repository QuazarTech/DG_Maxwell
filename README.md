# DG Maxwell


```
_____   _____   __  __                          _ _ 
|  __ \ / ____| |  \/  |                        | | |
| |  | | |  __  | \  / | __ ___  ____      _____| | |
| |  | | | |_ | | |\/| |/ _` \ \/ /\ \ /\ / / _ \ | |
| |__| | |__| | | |  | | (_| |>  <  \ V  V /  __/ | |
|_____/ \_____| |_|  |_|\__,_/_/\_\  \_/\_/ \___|_|_|
```

## Introduction
This projects aims at developing a fast Maxwell's Equation solver using
using the discontinuous Galerkin method. As first steps to develop this
solver, we are developing a library which allows us to solve the
1D wave equation and the 2D wave equation using discontinuous Galerkin
method.
>>>>>>> upstream-master

### 1D Wave equation
The 1D wave equation solver is aimed at finding the time evolution of
the 1D wave equation using the discontinuous Galerkin method. The
1D wave equation is given by the equation:

![1d_wave_eqn](./.svgs/1d_wave_eqn.svg )

where,

![u](./.svgs/u_1d.svg )

![F](./.svgs/F_1d.svg )

where, ![c](./.svgs/c_1d.svg ) is a number which denotes the wave
speed.

### 2D Wave Equation Solver
The 2D wave equation solver is aimed at finding the time evolution
of the 2D wave equation using the discontinuous Galerkin method.
The 2D wave equation is given by the equation:

![2d_wave_eqn](./.svgs/2d_wave_eqn.svg )

where,

![u](./.svgs/u.svg )

![F](./.svgs/F.svg )

![c](./.svgs/c.svg )

where, ![c_0](./.svgs/c_0.svg ) and ![c_1](./.svgs/c_1.svg ) denotes
the component of the wave speed in the ![hat_i](./.svgs/hat_i.svg )
and ![hat_j](./.svgs/hat_j.svg ) direction respectively.

Through the development of the 2D wave equation solver, the code and
the algorithms developed here will help in creating the 2D Maxwell's
equation solver. Currently the 2D wave equation solver will focus on
solving the wave equation in a rectangular domain in the
![xy](./.svgs/x_y.svg "x_y") plane, with periodic boundary conditions.

## Dependencies
- [matplotlib](https://matplotlib.org/)
- [numpy](http://www.numpy.org/)
- [arrayfire](http://arrayfire.org)
- [gmshtranslate](https://github.com/amanabt/gmshtranslator)
- [texlive](https://www.tug.org/texlive/)

## Documentation
The documenation for the project can be generated using `sphinx`
from the `docs` directory. You may read the instructions
[here](./docs/README.md).

## Unit tests
The unit tests are located in the
[code/tests](code/tests/) directory.

### Dependencies
- [pytest](https://docs.pytest.org/en/latest/#)

### Running Unit Tests
To run the unit tests, enter the following commands
```
$ cd path/to/the/DG_Maxwell/repo
$ pytest
```
* The parameters of the simulation are stored in params.py in
  the app folder, These can be changed accordingly.
  
* The images of the wave are stored in the `1D_wave_images` folder.

## Maintainers
- Aman Abhishek Tiwari - ![aman@quazartech.com](aman@quazartech.com)
- Balavarun P - ![Github Profile](https://github.com/balavarun5)
- Manichandra Morumpudi - ![mani@quazartech.com](mani@quazartech.com)
