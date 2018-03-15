# PPR
[![Travis Status](https://travis-ci.org/pavelkomarov/projection-pursuit.svg?branch=master)](https://travis-ci.org/pavelkomarov/projection-pursuit)
[![Coverage Status](https://coveralls.io/repos/github/pavelkomarov/projection-pursuit/badge.svg?branch=master&service=github)](https://coveralls.io/github/pavelkomarov/projection-pursuit?branch=master&service=github)
[![CircleCI Status](https://circleci.com/gh/pavelkomarov/projection-pursuit.svg?style=shield&circle-token=:circle-token)](https://circleci.com/gh/pavelkomarov/projection-pursuit/tree/master)

This repository is home to a couple [scikit-learn](http://scikit-learn.org/)-compatible estimators based on Jerome Friedman's generalizations[1] of his and Werner Stuetzle's *Projection Pursuit Regression* algorithm[2][3]. A regressor capable of multivariate estimation and a classifier based on regression to a one-hot multivariate representation are included.

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

For a few usage examples, see the examples and benchmarks directories. For an intuition of what the learner is doing, try running `viz_training_process.py`. For comparisons to other learners and an intuition of why you might want to try PPR, try the benchmarks. For a deep dive in to the math and an explanation of exactly how and why this works, see [`math.pdf`](https://github.com/pavelkomarov/projection-pursuit/blob/master/doc/math.pdf).

## References

1. Friedman, Jerome. (1985). "Classification and Multiple Regression Through Projection Pursuit." http://www.slac.stanford.edu/pubs/slacpubs/3750/slac-pub-3824.pdf
2. Hastie, Tibshirani, & Friedman. (2016). *The Elements of Statistical Learning 2nd Ed.*, section 11.2.
3. (2017) *Projection pursuit regression* https://en.wikipedia.org/wiki/Projection_pursuit_regression






TODO:

### 1. Make Sphinx output prettier and host somewhere

### 2. Setting up Circle CI
The project uses [CircleCI](https://circleci.com/) to build its documentation
from the `master` branch and host it using [Github Pages](https://pages.github.com/).
Again, you will need to Sign Up and authorize CircleCI. The configuration
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
