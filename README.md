# PPR
[![Travis Status](https://travis-ci.org/pavelkomarov/projection-pursuit.svg?branch=master)](https://travis-ci.org/pavelkomarov/projection-pursuit)
[![Coverage Status](https://coveralls.io/repos/github/pavelkomarov/projection-pursuit/badge.svg?branch=master)](https://coveralls.io/github/pavelkomarov/projection-pursuit?branch=master)
[![CircleCI Status](https://circleci.com/gh/pavelkomarov/projection-pursuit.svg?style=shield&circle-token=:circle-token)](https://circleci.com/gh/pavelkomarov/projection-pursuit/tree/master)

This repository is home to a couple [scikit-learn](http://scikit-learn.org/)-compatible estimators based on Jerome Friedman's generalizations[1] of his and Werner Stuetzle's *Projection Pursuit Regression* algorithm[2][3]. A regressor capable of multivariate estimation and a classifier based on regression to a one-hot multivariate representation are included.

## Installation and Usage
The package by itself comes with a single module containing the estimators. Before
installing the module you will need `numpy` and `scipy`.
To install the module execute:

```shell
pip install projection-pursuit
```
or
```shell
$ python setup.py install
``` 

If the installation is successful, and `scikit-learn` is correctly installed,
you should be able to execute the following in Python:
```python
>>> from skpp import ProjectionPursuitRegressor
>>> estimator = ProjectionPursuitRegressor()
>>> estimator.fit(np.arange(10).reshape(10, 1), np.arange(10))
```

## References

1. Friedman, Jerome. (1985). "Classification and Multiple Regression Through Projection Pursuit." http://www.slac.stanford.edu/pubs/slacpubs/3750/slac-pub-3824.pdf
2. Hastie, Tibshirani, & Friedman. (2016). *The Elements of Statistical Learning 2nd Ed.*, section 11.2.
3. (2017) *Projection pursuit regression* https://en.wikipedia.org/wiki/Projection_pursuit_regression



TODO:

### 1. Requirements
* standalone examples to illustrate the usage, model visualisation, and
  benefits/benchmarks of particular algorithms

### 2. Modifying the Documentation

The documentation is built using [sphinx](http://www.sphinx-doc.org/en/stable/).
It incorporates narrative documentation from the `doc/` directory, standalone
examples from the `examples/` directory, and API reference compiled from
estimator docstrings.

To build the documentation locally, ensure that you have `sphinx`,
`sphinx-gallery` and `matplotlib` by executing:
```shell
$ pip install sphinx matplotlib sphinx-gallery
```
The documentation contains a home page (`doc/index.rst`), an API
documentation page (`doc/api.rst`) and a page documenting the `template` module 
(`doc/template.rst`). Sphinx allows you to automatically document your modules
and classes by using the `autodoc` directive (see `template.rst`). To change the
asthetics of the docs and other paramteres, edit the `doc/conf.py` file. For
more information visit the [Sphinx Documentation](http://www.sphinx-doc.org/en/stable/contents.html).

You can also add code examples in the `examples` folder. All files inside
the folder of the form `plot_*.py` will be executed and their generated
plots will be available for viewing in the `/auto_examples` URL.

To build the documentation locally execute
```shell
$ cd doc
$ make html
```

### 3. Setting up Circle CI
The project uses [CircleCI](https://circleci.com/) to build its documentation
from the `master` branch and host it using [Github Pages](https://pages.github.com/).
Again,  you will need to Sign Up and authorize CircleCI. The configuration
of CircleCI is governed by the `circle.yml` file, which needs to be mofified
if you want to setup the docs on your own website. The values to be changed
are

| Variable | Value|
|----------|------|
| `USERNAME`  | The name of the user or organization of the repository where the project and documentation is hosted  |
| `DOC_REPO` | The repository where the documentation will be hosted. This can be the same as the project repository |
| `DOC_URL` | The relative URL where the documentation will be hosted |
| `EMAIL` | The email id to use while pushing the documentation, this can be any valid email address |

In addition to this, you will need to grant access to the CircleCI computers
to push to your documentation repository. To do this, visit the Project Settings
page of your project in CircleCI. Select `Checkout SSH keys` option and then
choose `Create and add user key` option. This should grant CircleCI privileges
to push to the repository `https://github.com/USERNAME/DOC_REPO/`.

If all goes well, you should be able to visit the documentation of your project
on 
```
https://github.com/USERNAME/DOC_REPO/DOC_URL
```
