import numpy
import pytest
import time

from sklearn.utils import estimator_checks
from sklearn.utils.testing import assert_equal
from sklearn.utils.testing import assert_less
from sklearn.utils.testing import assert_almost_equal
from sklearn.utils.testing import assert_array_less
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_array_almost_equal
from sklearn.utils.testing import assert_raise_message
from sklearn.utils.testing import assert_raises
from sklearn.utils.testing import assert_warns_message
# https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/utils/testing.py

from ..skpp import ProjectionPursuitRegressor, ProjectionPursuitClassifier

@pytest.mark.fast_test
def test_all_pass_sklearn_check_estimator():
	estimator_checks.MULTI_OUTPUT.append('ProjectionPursuitRegressor')
	estimator_checks.check_estimator(ProjectionPursuitRegressor)

@pytest.mark.fast_test
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

	estimator = ProjectionPursuitRegressor(r=20, fit_type='polyfit', degree=3,
		opt_level='high', weights='inverse-variance')

	print('Average magnitude of squared Y per element', numpy.sum(Y**2)/Y.size)

	print('training')
	before = time.time()
	estimator.fit(X[training, :], Y[training, :])
	after = time.time()
	print('finished in', after-before, 'seconds')

	Yhat = estimator.predict(X[training, :])
	train_error = numpy.sum((Y[training, :] - Yhat)**2)/Y[training, :].size
	print('Average magnitude of squared error in training data per element',
		train_error)

	Yhat = estimator.predict(X[testing, :])
	test_error = numpy.sum((Y[testing, :] - Yhat)**2)/Y[testing, :].size
	print('Average magnitude of squared error in testing data per element',
		test_error)

	assert_less(train_error, 1e-7)
	assert_less(test_error, 1e-7)