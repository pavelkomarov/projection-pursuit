import numpy
from scipy.interpolate import UnivariateSpline
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from matplotlib import pyplot

class MVPPR(BaseEstimator):
	def __init__(self, r=10, fit_type='polyfit', degree=3, opt_level='high',
				 eps_stage=0.001, eps_backfit=0.01, stage_maxiter=100,
				 backfit_maxiter=10):
		self.r = r
		self.fit_type = fit_type
		self.degree = degree
		self.opt_level = opt_level
		self.eps_stage = eps_stage
		self.eps_backfit = eps_backfit
		self.stage_maxiter = stage_maxiter
		self.backfit_maxiter = backfit_maxiter

	def predict(self, X):
		P = self.transform(X) # n x r
		# Take f_j of each projection (n x 1) and multiply by output weights
		# (1 x d), and sum over all  
		return sum([numpy.outer(self._f[j](P[:, j]), self._beta[:, j]) for j in range(self.r)])

	def transform(self, X):
		""" alpha is a p x r matrix [  |    |       |  ]
									[ a_0  a_1 ... a_r ]
									[  |    |       |  ]
			So dot(X, alpha) stores the projection of X through a_j
			in the jth column.
		"""
		check_is_fitted(self, '_alpha')
		X = check_array(X)
		return numpy.dot(X, self._alpha)

	def fit(self, X, Y):
		X, Y = check_X_y(X, Y, multi_output=True)
		if Y.ndim == 1:
			Y = Y.reshape((-1,1))

		self._alpha = numpy.random.randn(X.shape[1], self.r)
		self._beta = numpy.random.randn(Y.shape[1], self.r)
		self._f = [lambda x: x*0 for j in range(self.r)]
		self._df = [None for j in range(self.r)]

		for j in range(self.r): # for each term in the additive model
			self._fit_stage(X, Y, j, True)

			if self.opt_level == 'high':
				self._backfit(X, Y, j, True)
			elif self.opt_level == 'medium':
				self._backfit(X, Y, j, False)
			elif self.opt_level != 'low':
				raise ValueError('opt_level should be low, medium, or high.')
		return self

	def _fit_stage(self, X, Y, j, fit_weights):
		P = self.transform(X) # n x r
		R_j = Y - sum([numpy.outer(self._f[t](P[:, t]), self._beta[:, t].T) for t 
			in range(self.r) if t is not j]) # n x d

		n = 0 # iteration counter
		prev_loss = numpy.inf
		loss = 10**10 # HACK
		p_j = P[:,j] # n x 1, the jth column of the projections matrix
		while (abs(prev_loss - loss) > self.eps_stage and n < self.stage_maxiter):
			
			# find new optimal f to map from projections to weighted residuals
			w = numpy.ones(Y.shape[1])
			w_R_j = numpy.dot(R_j, w*self._beta[:, j]) / (numpy.inner(
				self._beta[:, j], w*self._beta[:, j]) + 1e-9)
			self._f[j], self._df[j] = self._fit_2d(p_j, w_R_j, j, n)
			
			# find new optimal beta vector to map from 1x1 function output to 1xd answer
			f = self._f[j](p_j) # n x 1
			self._beta[:, j] = numpy.dot(R_j.T, f) / (numpy.inner(f, f) + 1e-9)

			# find new optimal alpha to project input data
			if fit_weights:
				J = -(self._df[j](p_j)*X.T).T # multiply rows of X by the -derivative of the projection
				JTJ = numpy.dot(J.T, J)
				A = sum([w[k]*(self._beta[k, j]**2)*JTJ for k in range(Y.shape[1])])
				G_j = R_j - numpy.outer(self._f[j](p_j), self._beta[:, j].T)
				b = -sum([w[k]*self._beta[k, j]*numpy.dot(J.T, G_j[:, k]) for k in range(Y.shape[1])])

				delta = numpy.linalg.lstsq(A, b)[0]
				alpha = self._alpha[:, j] + delta # TODO implement halving step
				self._alpha[:, j] = alpha/numpy.linalg.norm(alpha) # normalize to avoid numerical drift

			# Recalculate projection with new beta_j and f_j
			p_j = numpy.dot(X, self._alpha[:, j])
			
			# Calculate mean squared error for this iteration
			prev_loss = loss
			diff = R_j - numpy.outer(self._f[j](p_j), self._beta[:, j].T) # subtract new prediction
			loss = numpy.sum((w*diff)**2)
			n += 1

	def _backfit(self, X, Y, j, fit_weights):
		n = 0
		prev_err = numpy.inf
		err = 0
		while (abs(prev_err - err) > self.eps_backfit and n < self.backfit_maxiter):
			for t in range(j):
				self._fit_stage(X, y, k, t, fit_weights)

			P_k = numpy.dot(X, self.beta[k])

			prev_err = err
			diff = y - sum([self._f[k][t](P_k[:,t]) for t in range(j)])
			err = numpy.inner(diff, diff)/len(diff) # fast calculate MSE
			n += 1

	def _fit_2d(self, x, y, j, itr):
		if self.fit_type == 'polyfit':
			coeffs = numpy.polyfit(x, y, deg=self.degree)
			fit = numpy.poly1d(coeffs)
			deriv = fit.deriv(m=1)
		elif self.fit_type == 'spline':
			order = numpy.argsort(x)
			# set s according to the recommendation at: stackoverflow.com/
			# questions/8719754/scipy-interpolate-univariatespline-not-
			# smoothing-regardless-of-parameters
			fit = UnivariateSpline(x[order], y[order], k=self.degree,
				s=len(residual)*residual.var(), ext=0)
			deriv = fit.derivative(1)
		else:
			raise ValueError(self.fit_type + ' is not a valid fit_type.')

		if False and (itr % 10 == 0):
			pyplot.scatter(x, y)
			pyplot.title('plot' + str(itr) + 'stage' + str(j))
			pyplot.xlabel('projections')
			pyplot.ylabel('residuals')
			xx = numpy.linspace(min(x), max(x), 100)
			yy = fit(xx)
			pyplot.plot(xx, yy, 'g', linewidth=1)
			pyplot.show()

		return fit, deriv
