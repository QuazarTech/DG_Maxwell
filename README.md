
# DG Maxwell

[![Codacy Badge](https://api.codacy.com/project/badge/Grade/8c3103477eb74f1a9d5a87c6b59c220f)](https://www.codacy.com/app/aman2official/DG_Maxwell?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=amanabt/DG_Maxwell&amp;utm_campaign=Badge_Grade)
[![Build Status](https://travis-ci.org/amanabt/DG_Maxwell.svg?branch=2D_wave_travis)](https://travis-ci.org/amanabt/DG_Maxwell)

```
_____   _____   __  __                          _ _ 
|  __ \ / ____| |  \/  |                        | | |
| |  | | |  __  | \  / | __ ___  ____      _____| | |
| |  | | | |_ | | |\/| |/ _` \ \/ /\ \ /\ / / _ \ | |
| |__| | |__| | | |  | | (_| |>  <  \ V  V /  __/ | |
|_____/ \_____| |_|  |_|\__,_/_/\_\  \_/\_/ \___|_|_|
```

## Introduction
This project is a stepping stone to develop a solver for solving
the 2D Maxwell's equations using the Discontinuous Galerkin method.
The project to develop 2D wave equation solver is a stepping stone
to develop the Maxwell's equation solver. The 2D wave equation is
given by the equation:

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
[2d_solver/tests](https://github.com/amanabt/DG_Maxwell/edit/2d_wave_solver/2d_solver/tests)
directory.

### Dependencies
- [pytest](https://docs.pytest.org/en/latest/#)

### Running Unit Tests
To run the unit tests, enter the following commands
```
$ cd path/to/the/DG_Maxwell/repo
$ pytest
```


## Maintainers
- Aman Abhishek Tiwari - ![aman@quazartech.com](aman@quazartech.com)
- Balavarun P - ![Github Profile](https://github.com/balavarun5)
- Manichandra Morumpudi - ![mani@quazartech.com](mani@quazartech.com)
