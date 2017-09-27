# DG Maxwell's Documentation

This is a `Sphinx` generated documentation for DG Maxwell code.

## Dependencies

- [sphinx](http://www.sphinx-doc.org/en/stable/install.html)
- [numpydoc](https://pypi.python.org/pypi/numpydoc)

#### Installation of Dependencies
You may download `sphinx` and `numpydoc` from official Ubuntu repository.
Enter these commands to install `sphinx` and `numpydoc`.

```
$ apt-get install python3-sphinx python3-numpydoc
```

> **Caution!**
  If you are installing `sphinx` and `numpydoc` from some other source,
  the `Python3` version of `sphinx` and `numpydoc` should be installed.


## Usage

To generate HTML documentation, open the terminal and enter the following commands
```
$ cd <path/to/DG_Maxwell/directory>
$ cd docs
$ make html
```

This will generate a documentation in the `_build` directory.
To access the documentation, change the directory to `_build/html`

```
$ cd _build/html
```

Now open the file `index.html` in this directory in a web browser to access the documentation.
