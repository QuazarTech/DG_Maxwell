# Contributing to DG Maxwell
You can contribute to this project by testing, raising issues
and making code contributions. A certain set of guidelines needs
to be followed for cotributing to the project.

## Code Guidelines
This project is mostly written in `Python3`, we follow a certain
set of code guidelines for writing `Python3` code. This project
follows the code guidelines set by the master organization
[Quazar Technologies](https://github.com/quazartech). You may
find the latest code guidelines on this
[link](https://github.com/QuazarTech/Style-Guidelines/blob/master/code_guidelines/python.md).

## Documentation Guidelines
This project uses [Sphinx](http://www.sphinx-doc.org/en/stable/)
to build it's documentation. The documentation lies in the
`docs` directory of the repository. In this project,
python `docstrings` are used to document all the member functions
of a module. Sphinx uses `autodoc` to read this docstring and
convert it into beautiful `html` pages. We follow a
certain set of guidelines for writing the `docstrings` as explained
below.

- Use `reST` syntax to write the documentation. Refer to these links,
  [reST](http://www.sphinx-doc.org/en/stable/rest.html#rst-primer) and
  [Sphinx Markup Constructs](http://www.sphinx-doc.org/en/stable/markup/index.html#sphinxmarkup)
  to write the documentation for `Sphinx`.
- This project uses `numpydoc` for it's documentation. See how to use it
  on this [link](https://github.com/numpy/numpy/blob/master/doc/HOWTO_DOCUMENT.rst.txt).
- Give a short description of what the function does.
- Mathematical variables should be written in LaTeX. See
  [Math support in Sphinx](http://www.sphinx-doc.org/en/stable/ext/math.html#math-support).
- When using the special LaTeX symbols, use double "\\\\" instead of a
    single "\\". For eg, for writing the math symbol
    ![xi](.svgs/xi.svg), write "\\\\xi" instead of "\\xi".
- If the function implements some mathematical equation, state that
    equation and describe the terms occuring in the equation.
- If the algorithm is non-trivial, explain it in the documentation.
- If the algorithm is very lengthy and complicated,
    write a LaTeX file explaining the algorithm and state it in the documentation.
    Here is an example for stating the link `Full description of the algorithm can
    be found here < link >`
- In case you want to insert the URL in the documentation and the URL is
    exceeding the 80 characters limit, shorten the URL using
    the [Google URL shortener](https://goo.gl/).
- Don't use personal pronouns in the documentation.
- Do not exceed the 80 characters per line limit.

Shown below is an example for documentation of a function
```python
def isoparam_1D(x_nodes, xi):
    '''
    Maps points in :math:`\\xi` space to :math:`x` space using the formula
    :math:`x = \\frac{1 - \\xi}{2} x_0 + \\frac{1 + \\xi}{2} x_1`

    Parameters
    ----------
    
    x_nodes : arrayfire.Array [2 1 1 1]
              Element nodes.
    
    xi      : arrayfire.Array [N 1 1 1]
              Value of :math:`\\xi` coordinate for which the corresponding
              :math:`x` coordinate is to be found.
    
    Returns
    -------
    x : arrayfire.Array
        :math:`x` value in the element corresponding to :math:`\\xi`.
    '''    '''
```

## Documentation Hosting
The documentations for the code is hosted on
[readthedocs(rtd)](https://readthedocs.org). This platform uses the `makefile`
in the `docs` directory to build a documentation and host it on the web if the
documentation build is successful. So, it's necessary to always make sure that
rtd is able to build the documentation successfully. Here are some things to
keep in mind when using rtd to host the documentation.
1.This project uses Sphinx `autodoc` extension to generate
  the documentation from the `doc-strings`. But, in doing so, it imports the
  actual modules. Those modules often contain some other 3rd party imports.
  Sphinx fails if that dependency is not satisfied. So, to avoid that situation,
  this project uses `unittest.mock.MagicMock`. This was you can use mock imports.
  This code snippet is added in `docs/conf.py`.
  ```python
  from unittest.mock import MagicMock

  class Mock(MagicMock):
      @classmethod
      def __getattr__(cls, name):
              return MagicMock()

  MOCK_MODULES = ['gmshtranslator', 'gmshtranslator.gmshtranslator', 'arrayfire',
                  'lagrange', 'scipy', 'numpy', 'matplotlib', 'matplotlib.lines',
                  'dg_maxwell.params', 'tqdm']
  sys.modules.update((mod_name, Mock()) for mod_name in MOCK_MODULES)
  ```
  If you want to add a mock import, add that import name in the `MOCK_MODULES`
  list.

## Continuous Integration
This project uses [Travis CI](https://travis-ci.org/) for automatically checking
the `build` and the unit tests.

## Code Quality
This project uses [CODACY](https://www.codacy.com/) to automatically check the
code quality.

## Pull Request Process
This project follows the guidelines set by Quazar Technologies for giving
a pull request. You may find the link to the pull request
[here](https://github.com/QuazarTech/Style-Guidelines/blob/master/github/PR_guidelines.md).
Here are few additional guidelines to be followed when giving a pull request
- Before asking to merge the pull request, make sure that the documentation
  build is passing on `rtd`. You may do this by hosting your fork documentation on
  `rtd`. Make sure that there are no errors by checking the build log and also
  check your documentation on the `rtd` website and make sure to check the
  hosted documentation of your fork.
- Make sure that the build is successfull and all the unit tests are passing on
  `Travis CI`.
- Check the code quality of your fork with CODACY.
