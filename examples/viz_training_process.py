"""This script can help you visualize the training process.

A plot is presented in the midst of the function-fitting step of each iteration
of the alternating optimization loop. It displays the residual variance (targets)
vs the projected data. There is no backfitting so each "stage" is fit exactly
once. Notice how the points draw closer to the line each iteration and how through
the stages, as more of the output variance is accounted for, the scale on the
Y axis shrinks.
"""

import numpy
import sys
sys.path.append("..")
from skpp import ProjectionPursuitRegressor

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

print('Average magnitude of squared Y per element', numpy.sum(Y**2)/Y.size)

ppr = ProjectionPursuitRegressor(opt_level='low', show_plots=True, plot_epoch=1)
ppr.fit(X, Y)

Yhat = ppr.predict(X)
error = numpy.sum((Y - Yhat)**2)/Y.size
print('Average magnitude of squared difference between predicted and original Y', error)
