# Projection Pursuit
[![Build Status](https://github.com/pavelkomarov/projection-pursuit/actions/workflows/build.yml/badge.svg)](https://github.com/pavelkomarov/projection-pursuit/actions)
[![Coverage Status](https://coveralls.io/repos/github/pavelkomarov/projection-pursuit/badge.svg?branch=master&service=github)](https://coveralls.io/github/pavelkomarov/projection-pursuit?branch=master&service=github)
[![Downloads](https://pepy.tech/badge/projection-pursuit)](https://pepy.tech/project/projection-pursuit)

[Documentation](https://pavelkomarov.com/projection-pursuit/skpp.html), [How it works](https://pavelkomarov.com/projection-pursuit/math.pdf).

This repository is home to a couple [scikit-learn](http://scikit-learn.org/)-compatible estimators based on Jerome Friedman's generalizations[1] of his and Werner Stuetzle's *Projection Pursuit Regression* algorithm[2][3]. A regressor capable of multivariate estimation and dimensionality reduction and a univariate classifier based on regression to a one-hot multivariate representation are included.

This repository is also meant to serve as a fairly pared-down example of how to use Github Actions, Coveralls, Sphinx, PyTest, how to deploy to PyPI and Github Pages, and how to create a Scikit-Learn Estimator that passes the sklearn checks and follows the PEP 8 style standard.

## Installation and Usage
The package by itself comes with a single module containing the estimators. Before
installing the module you will need `numpy`, `scipy`, `scikit-learn`, and `matplotlib`.
To install the module execute:

```shell
pip install projection-pursuit
```
or
```shell
$ python setup.py install
``` 

If the installation is successful, you should be able to execute the following in Python:
```python
>>> from skpp import ProjectionPursuitRegressor
>>> estimator = ProjectionPursuitRegressor()
>>> estimator.fit(np.arange(10).reshape(10, 1), np.arange(10))
```

Sphinx is run via continuous integration to generate [the API](https://pavelkomarov.com/projection-pursuit/skpp.html).

For a few usage examples, see the examples and benchmarks directories. For an intuition of what the learner is doing, try running `viz_training_process.py`. For comparisons to other learners and an intuition of why you might want to try PPR, try the benchmarks. For a deep dive in to the math and an explanation of exactly how and why this works, see [`math.pdf`](https://pavelkomarov.com/projection-pursuit/math.pdf).

## References

1. Friedman, Jerome. (1985). "Classification and Multiple Regression Through Projection Pursuit." http://www.slac.stanford.edu/pubs/slacpubs/3750/slac-pub-3824.pdf
2. Hastie, Tibshirani, & Friedman. (2016). *The Elements of Statistical Learning 2nd Ed.*, section 11.2.
3. (2017) *Projection pursuit regression* https://en.wikipedia.org/wiki/Projection_pursuit_regression
