# DG Maxwell

[![Build Status](https://travis-ci.org/QuazarTech/DG_Maxwell.svg?branch=master)](https://travis-ci.org/QuazarTech/DG_Maxwell)
[![Codacy Badge](https://api.codacy.com/project/badge/Grade/e8733cdbf1454af0ac35ae5b2d017d9f)](https://www.codacy.com/app/aman2official/DG_Maxwell_2?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=QuazarTech/DG_Maxwell&amp;utm_campaign=Badge_Grade)
[![Documentation Status](http://readthedocs.org/projects/dg-maxwell/badge/?version=latest)](http://dg-maxwell.readthedocs.io/en/latest/?badge=latest)
[![Coverage Status](https://coveralls.io/repos/github/QuazarTech/DG_Maxwell/badge.svg?branch=master)](https://coveralls.io/github/QuazarTech/DG_Maxwell?branch=master)
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
- [coveralls](https://pypi.python.org/pypi/coveralls)
- [python-coveralls](https://pypi.python.org/pypi/python-coveralls/)

## Documentation
The documenation for the project can be generated using `sphinx`
from the `docs` directory. You may read the instructions
[here](./docs/README.md).

## Unit tests
The unit tests are located in the
[code/tests](code/tests/) directory.

### Dependencies
- [pytest](https://docs.pytest.org/en/latest/#)
- [pytest-cov](https://pypi.python.org/pypi/pytest-cov)
### Running Unit Tests
To run the unit tests, enter the following commands
```
$ cd path/to/the/DG_Maxwell/repo
$ pytest --verbose -r P --color=yes --cov dg_maxwell
```
* The parameters of the simulation are stored in params.py in
  the app folder, These can be changed accordingly.
  
* To obtain the movie of the 1D wave advection, run `post_processing.py` file after the simulation.
  The movie will be created automatically and stored in `results` folder.

## Maintainers
- Aman Abhishek Tiwari - ![aman@quazartech.com](aman@quazartech.com)
- Balavarun P - ![f2013462@pilani.bits-pilani.ac.in](f2013462@pilani.bits-pilani.ac.in)
- Manichandra Morumpudi - ![mani@quazartech.com](mani@quazartech.com)
