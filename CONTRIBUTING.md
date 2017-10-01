# Contributing to DG Maxwell
You can contribute to this project by testing, raising issues
and making code contributions. A certain set of guidelines are
required to be followed for cotributing to the project in
different ways.

## Code Guidelines
This project is mostly written in `Python3`, we follow a certain
set of code guidelines for writing `python3` code. This project
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
convert it into a beautiful `html` documentation. We follow a
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
    ![xi](.svgs/xi.svg), write "$\\\\xi$" instead of "$\\xi$".
- If the function implements some mathematical equation, state that
    equation and describe the terms occuring in the equation.
- If the algorithm is non-trivial, explain it in the documentation.
- If the algorithm is very lengthy and complicated,
    write a LaTeX file explaining the algorithm and state it in the documentation.
    Here is an example for stating the link "Full description of the algorithm can
    be found here < link >"
- In case you want to insert the URL in the documentation and the URL is
    exceeding the 80 characters limit, you have to shorten the URL using
    the [Google URL shortener](https://goo.gl/).
- Don't use personal pronouns in the documentation.
- Do not exceed the 80 characters per line limit.

Shown below is an example for documentation of a function
```
    '''
    Finds the :math:`x` coordinate using isoparametric mapping of a
    :math:`2^{nd}` order element with :math:`8` nodes
    .. math:: (P_0, P_1, P_2, P_3, P_4, P_5, P_6, P_7)

    Here :math:`P_i` corresponds to :math:`(\\xi_i, \\eta_i)` coordinates,
    :math:`i \in \\{0, 1, ..., 7\\}` respectively, where,
    
    .. math:: (\\xi_0, \\eta_0) &\equiv (-1,  1) \\\\
              (\\xi_1, \\eta_1) &\equiv (-1,  0) \\\\
              (\\xi_2, \\eta_2) &\equiv (-1, -1) \\\\
              (\\xi_3, \\eta_3) &\equiv ( 0, -1) \\\\
              (\\xi_4, \\eta_4) &\equiv ( 1, -1) \\\\
              (\\xi_5, \\eta_5) &\equiv ( 1,  0) \\\\
              (\\xi_6, \\eta_6) &\equiv ( 1,  1) \\\\
              (\\xi_7, \\eta_7) &\equiv ( 0,  1)
              
    Parameters
    ----------
    x_nodes : np.ndarray [8]
              :math:`x` nodes.
              
    xi      : float
            :math:`\\xi` coordinate for which :math:`x` has to be found.

    eta     : float
            :math:`\\eta` coordinate for which :math:`x` has to be found.

    Returns
    -------
    x : float
        :math:`x` coordinate corresponding to :math:`(\\xi, \\eta)` coordinate.
    
    '''
```

