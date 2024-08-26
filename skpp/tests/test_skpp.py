# run with python(3) -m pytest

import numpy
import pytest
import time

from sklearn.utils import estimator_checks
from sklearn.utils._testing import assert_raises

from ..skpp import ProjectionPursuitRegressor, ProjectionPursuitClassifier

def test_regressor_passes_sklearn_checks():
	estimator_checks.check_estimator(ProjectionPursuitRegressor())

def test_classifier_passes_sklearn_checks(): # Note this one causes a warning in the single-class case
	estimator_checks.check_estimator(ProjectionPursuitClassifier())

def test_construction_errors():
	assert_raises(ValueError, ProjectionPursuitRegressor, r=0)
	assert_raises(NotImplementedError, ProjectionPursuitRegressor, fit_type='jabberwocky')
	assert_raises(ValueError, ProjectionPursuitRegressor, degree='master')
	assert_raises(ValueError, ProjectionPursuitRegressor, opt_level='near')
	assert_raises(ValueError, ProjectionPursuitRegressor, example_weights='light')
	assert_raises(ValueError, ProjectionPursuitRegressor, example_weights=numpy.array([-1]))
	assert_raises(ValueError, ProjectionPursuitRegressor, out_dim_weights='heavy')
	assert_raises(ValueError, ProjectionPursuitRegressor, out_dim_weights=numpy.array([-1]))
	assert_raises(ValueError, ProjectionPursuitRegressor, eps_stage=-0.1)
	assert_raises(ValueError, ProjectionPursuitRegressor, stage_maxiter=0)
	assert_raises(ValueError, ProjectionPursuitClassifier, pairwise_loss_matrix=numpy.array([-1]))
	assert_raises(ValueError, ProjectionPursuitClassifier, pairwise_loss_matrix=numpy.array([1]))
	assert_raises(ValueError, ProjectionPursuitClassifier, pairwise_loss_matrix='whereami?')

def test_fit_errors():
	ppc = ProjectionPursuitClassifier(example_weights=numpy.array([1, 2]))
	ppr = ProjectionPursuitRegressor(example_weights=numpy.array([1,2]),
		out_dim_weights=numpy.array([3]))
	X = numpy.random.randn(5, 2)
	Y = numpy.array([0, 0, 1, 1, 1])
	assert_raises(ValueError, ppc.fit, X, Y)
	assert_raises(ValueError, ppr.fit, X, Y)
	X = numpy.random.randn(2, 2)
	Y = numpy.eye(2)
	assert_raises(ValueError, ppc.fit, X, Y)
	assert_raises(ValueError, ppr.fit, X, Y)

def test_example_weightings_applied():
	# Construct a 1D example constrained to deg=2. No polynomial of such low
	# order can go through all the points, so weights determine which should be
	# fit more closely.
	X = numpy.array([[-1],[-0.9],[0],[0.9],[1]])# on a number line
	Y = numpy.array([0, 1, 1, 1, 0])# the targets for these points

	L = numpy.array([[0, 10], [1, 0]])

	# If given the following example weightings, the rounded predictions at the
	# points queried should end up looking like the corresponding targets.
	example_weights = numpy.array([[1, 1, 1, 1, 1], [1, 100, 100, 100, 1],
		[100, 1, 1, 1, 100], [10, 1, 1, 10, 1]])
	targets = numpy.array([[0, 0, 1, 1, 1, 0, 0], [1, 1, 1, 1, 1, 1, 1],
		[0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1, 1]])

	for i in range(example_weights.shape[0]):
		ppr = ProjectionPursuitRegressor(degree=2,
			example_weights=example_weights[i,:])
		ppr.fit(X, Y)

		ppc = ProjectionPursuitClassifier(degree=2,
			example_weights=example_weights[i,:], pairwise_loss_matrix=L)
		ppc.fit(X, Y)

		predictions = numpy.round(ppr.predict(numpy.array([[-1], [-0.95], [-0.9],
			[0], [0.9], [0.95], [1]])))

		assert numpy.array_equal(predictions, targets[i,:])

def test_ppr_learns():
	# Generate some dummy data, X random, Y an additive-model-like construction
	n = 1000
	d = 4
	p = 10

	X = numpy.random.rand(n, p) - 0.5
	Y = numpy.zeros((n, d))
	for j in range(5):
		alpha = numpy.random.randn(p) # projection vector
		projection = numpy.dot(X, alpha)
		# Generate random polynomials with coefficients in [-100, 100]
		f = numpy.poly1d(numpy.random.randint(-100, 100,
			size=numpy.random.randint(3+1)))
		beta = numpy.random.randn(d) # expansion vector
		Y += numpy.outer(f(projection), beta)

	# Divide the data
	temp = numpy.arange(n)
	numpy.random.shuffle(temp)
	training = temp[0:int(n*0.8)]
	testing = temp[int(n*0.8):]

	mse_per_element = numpy.sum(Y**2)/Y.size
	print('Average magnitude of squared Y per element', mse_per_element)

	estimators = [ProjectionPursuitRegressor(r=20, fit_type='polyfit', degree=3,
		opt_level='high'), ProjectionPursuitRegressor(out_dim_weights='uniform',
		fit_type='spline', opt_level='medium')]
	accuracy_thresholds = [mse_per_element/1000000, mse_per_element/100] # targets to meet

	for i in range(len(estimators)):
		
		print('training')
		before = time.time()
		estimators[i].fit(X[training, :], Y[training, :])
		after = time.time()
		print('finished in', after-before, 'seconds')

		Yhat = estimators[i].predict(X[training, :])
		train_error = numpy.sum((Y[training, :] - Yhat)**2)/Y[training, :].size
		print('Average magnitude of squared error in training data per element',
			train_error)

		Yhat = estimators[i].predict(X[testing, :])
		test_error = numpy.sum((Y[testing, :] - Yhat)**2)/Y[testing, :].size
		print('Average magnitude of squared error in testing data per element',
			test_error)

		assert train_error < accuracy_thresholds[i]
		assert test_error < accuracy_thresholds[i]
