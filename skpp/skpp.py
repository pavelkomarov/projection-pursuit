import numpy
from scipy.interpolate import UnivariateSpline
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin, \
	ClassifierMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_X_y, check_is_fitted
from sklearn.utils.multiclass import unique_labels
from sklearn.utils import as_float_array, check_random_state, check_array
from matplotlib import pyplot
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

class ProjectionPursuitRegressor(BaseEstimator, TransformerMixin, RegressorMixin):
	""" This class implements the PPR algorithm as detailed in math.pdf.

	Parameters
	----------
	`r` int, default=10:
		The number of terms in the underlying additive model. The input will be
		put through `r` projections, `r` functions of those projections, and
		then multiplication by `r` output vectors to determine output.

	`fit_type` {'polyfit', 'spline'}, default='polyfit':
		The kind of function to fit at each stage.

	`degree` int, default=3:
		The degree of polynomials or spline-sections used as the univariate
		approximator between projection and .

	`opt_level` {'high', 'medium', 'low'}, default='high':
		'low' opt_level will disable backfitting. 'medium' backfits previous
		2D functional fits only (not projections). 'high' backfits everything.

	'example_weights' string or array-like of dimension (n_samples,), default='uniform':
		The relative importances given to training examples when calculating
		loss and solving for parameters.

	'out_dim_weights' string or array-like, default='inverse-variance':
		The relative importances given to output dimensions when calculating the
		weighted residual (output of the univariate functions f_j). If all
		dimensions are of the same importance, but outputs are of different
		scales, then using the inverse variance is a good choice.
		Possible values:
			`'inverse-variance'`: Divide outputs by their variances.
			`'uniform'`: Use a vector of ones as the weights.
			`array`: Provide a custom vector of weights of dimension (n_outputs,)

	`eps_stage` float, default=0.0001:
		The mean squared difference between the predictions of the PPR at
		subsequent iterations of a "stage" (fitting an f, beta pair) must reach
		below this epsilon in order for the stage to be considered converged.

	`eps_backfit` float, default=0.01:
		The mean squared difference between the predictions of the PPR at
		subsequent iterations of a "backfit" must reach below this epsilon in
		order for backfitting to be considered converged.

	`stage_maxiter` int, default=100:
		If a stage does not converge within this many iterations, end the loop
		and move on. This is useful for divergent cases.

	`backfit_maxiter` int, default=10:
		If a backfit does not converge withint this many iterations, end the
		loop and move on. Smaller values may be preferred here since backfit
		iterations are expensive.

	`random_state` int, numpy.RandomState, default=None:
		An optional object with which to seed randomness.

	`show_plots` boolean, default=False:
		Whether to produce plots of projections versus residual variance
		throughout the training process.

	`plot_epoch` int, default=50:
		If plots are displayed, show them every `plot_epoch` iterations of the
		stage-fitting process.
	"""
	def __init__(self, r=10, fit_type='polyfit', degree=3, opt_level='high',
				 example_weights='uniform', out_dim_weights='inverse-variance',
				 eps_stage=0.0001, eps_backfit=0.01, stage_maxiter=100,
				 backfit_maxiter=10, random_state=None, show_plots=False,
				 plot_epoch=50):

		# paranoid parameter checking to make it easier for users to know when
		# they have gone awry and to make it safe to assume some variables can
		# only have certain settings
		if not isinstance(r, int) or r < 1:
			raise ValueError('r must be an int >= 1.')
		if fit_type not in ['polyfit', 'spline']:
			raise NotImplementedError('fit_type ' + fit_type + ' not supported')
		if not isinstance(degree, int) or degree < 1:
			raise ValueError('degree must be >= 1.')
		if opt_level not in ['low', 'medium', 'high']:
			raise ValueError('opt_level must be either low, medium, or high.')
		if not (isinstance(example_weights, str) and example_weights == 'uniform'):
			try:
				example_weights = as_float_array(example_weights)
			except (TypeError, ValueError) as error:
				raise ValueError('example_weights must be `uniform`, or array-like.')
			if numpy.any(example_weights < 0):
				raise ValueError('example_weights can not contain negatives.')
		if not (isinstance(out_dim_weights, str) and out_dim_weights in
			['inverse-variance', 'uniform']):
			try:
				out_dim_weights = as_float_array(out_dim_weights)
			except (TypeError, ValueError) as error:
				raise ValueError('out_dim_weights must be either ' + \
					'inverse-variance, uniform, or array-like.')
			if numpy.any(out_dim_weights < 0):
				raise ValueError('out_dim_weights can not contain negatives.')
		if eps_stage <= 0 or eps_backfit <= 0:
			raise ValueError('Epsilons must be > 0.')
		if not isinstance(stage_maxiter, int) or stage_maxiter <= 0 or \
			not isinstance(backfit_maxiter, int) or backfit_maxiter <= 0:
			raise ValueError('Maximum iteration settings must be ints > 0.')

		# Save parameters to the object
		params = locals()
		for k, v in params.items():
			if k != 'self':
				setattr(self, k, v)

	def transform(self, X):
		""" Find the projections of X through all alpha vectors in the PPR.

			_alpha is a p x r matrix [  |    |        |   ]
									 [ a_0  a_1 ... a_r-1 ]
									 [  |    |        |   ]
			and X is an n x p matrix [ ---x_0--- ]
									 [ ---x_1--- ]
									 [    ...    ]
									 [ --x_n-1-- ]
			So the inner X with _alpha stores the projections of X through
			alpha_j in the jth column of the result:
			P = [  x_0*a_0   x_0*a_1  ...  x_0*a_r-1  ]
				[  x_1*a_0   x_1*a_1  ...  x_1*a_r-1  ]
				[   ...       ...     ...      ...    ]
				[ x_n-1*a_0 x_n-1*a_1 ... x_n-1*a_r-1 ]

		Parameters
		----------
		`X` array-like of shape (n_samples, n_features):
			The input samples.

		Returns
		-------
		Projections, an array of shape (n_inputs, r):
			where r is the hyperparameter given to the constructor, the number
			of terms in the additive model, and the jth column is the projection
			of X through alpha_j.
		"""
		check_is_fitted(self, '_alpha')
		X = check_array(X)
		return numpy.dot(X, self._alpha)

	def predict(self, X):
		""" Use the fitted estimator to make predictions on new data.

		Parameters
		----------
		`X` array-like of shape = (n_samples, n_features):
			The input samples.

		Returns
		-------
		`Y` array of shape = (n_samples) or (n_samples, n_outputs):
			The result of passing X through the evaluation function.
		"""
		# Check whether the PPR is trained, and if so get the projections.
		P = self.transform(X) # P is an n x r matrix.
		# Take f_j of each projection, yielding a vector of shape (n,); take the
		# outer product with the corresponding output weights of shape (d,); and
		# take the weighted sum over all terms. This is a vectorized version of
		# the evaluation function. 
		Y = sum([numpy.outer(self._f[j](P[:, j]), self._beta[:, j])
			for j in range(self.r)])
		# return single-dimensional output if Y has only one column
		return Y if Y.shape[1] != 1 else Y[:,0]

	def fit(self, X, Y):
		""" Train the model.

		Parameters
		----------
		`X` array-like of shape = (n_samples, n_features):
			The training input samples.
		`Y` array-like, shape = (n_samples,) or (n_samples, n_outputs)
			The target values.

		Returns
		-------
		self ProjectionPursuitRegressor:
			A trained model.
		"""
		X, Y = check_X_y(X, Y, multi_output=True)
		if Y.ndim == 1: # standardize Y as 2D so the below always works
			Y = Y.reshape((-1,1)) # reshape returns a view to existing data

		self._random = check_random_state(self.random_state)

		# Sklearn does not allow mutation of object parameters (the ones not
		# prepended by an underscore), so construct or reassign weights
		if isinstance(self.example_weights, str) and \
			self.example_weights == 'uniform':
			self._example_weights = numpy.ones(X.shape[0])
		elif isinstance(self.example_weights, numpy.ndarray):
			if X.shape[0] != self.example_weights.shape[0]:
				raise ValueError('example_weights provided to the constructor' +
					' have dimension ' + str(self.example_weights.shape[0]) +
					', which disagrees with the size of X: ' + str(X.shape[0]))
			else:
				self._example_weights = self.example_weights

		if isinstance(self.out_dim_weights, str) and \
			self.out_dim_weights == 'inverse-variance':
			variances = Y.var(axis=0)
			if max(variances) == 0:
				raise ValueError('Y must have some variance.')
			# There is a problem if a variance for any column is zero, because
			# its relative scale will appear infinite. Fill zeros with the max
			# of variances, so the corresponding columns have small weight and
			# are not major determiners of loss.
			variances[variances == 0] = max(variances)
			self._out_dim_weights = 1./variances
		elif isinstance(self.out_dim_weights, str) and \
			self.out_dim_weights == 'uniform':
			self._out_dim_weights = numpy.ones(Y.shape[1])
		elif isinstance(self.out_dim_weights, numpy.ndarray):
			if Y.shape[1] != self.out_dim_weights.shape[0]:
				raise ValueError('out_dim_weights provided to the constructor' +
					' have dimension ' + str(self.out_dim_weights.shape[0]) +
					', which disagrees with the width of Y: ' + str(Y.shape[1]))
			else:
				self._out_dim_weights = self.out_dim_weights

		# Now that input and output dimensions are known, parameters vectors
		# can be initialized. Vectors are always stored vertically.
		self._alpha = self._random.randn(X.shape[1], self.r) # p x r
		self._beta = self._random.randn(Y.shape[1], self.r) # d x r
		self._f = [lambda x: x*0 for j in range(self.r)] # zero functions
		self._df = [None for j in range(self.r)] # no derivatives yet

		for j in range(self.r): # for each term in the additive model
			self._fit_stage(X, Y, j, True)

			if self.opt_level == 'high':
				self._backfit(X, Y, j, True)
			elif self.opt_level == 'medium':
				self._backfit(X, Y, j, False)
		return self

	def _fit_stage(self, X, Y, j, fit_weights):
		""" A "stage" consists of a set of alpha_j, f_j, and beta_j parameters.
		Given the stages already fit, find the residual this stage should try
		to match, and perform alternating optimization until parameters have
		converged.

		Parameters
		----------
		`X` array-like of shape = (n_samples, n_features):
			The training input samples.
		`Y` array-like, shape (n_samples, n_outputs)
			The target values.
		`j` int:
			The index of this stage in the additive model.
		`fit_weights` boolean:
			Whether to refit alpha_j or leave it unmodified.
		"""
		# projections P = X*Alphas, P_j = X*alpha_j
		P = self.transform(X) # the n x r projections matrix
		# The residuals matrix is essentially the evaluation function separated
		# in to the contribution of this term vs all the others and then
		# algebraically solved for this term's contribution by subtracting the
		# rest of the sum from both sides.
		R_j = Y - sum([numpy.outer(self._f[t](P[:, t]), self._beta[:, t].T) for t 
			in range(self.r) if t is not j]) # the n x d residuals matrix

		# main alternating optimization loop
		itr = 0 # iteration counter
		# Start off with dummy infinite losses to get the loop started because
		# no value of loss should be able to accidentally fall within epsilon of
		# a dummy value and cause the loop to prematurely terminate.
		prev_loss = -numpy.inf
		loss = numpy.inf
		p_j = P[:,j] # n x 1, the jth column of the projections matrix
		# Use the absolute value between loss and previous loss because we do
		# not want to terminate if the instability of parameters in the first
		# few iterations causes loss to momentarily increase.
		while (abs(prev_loss - loss) > self.eps_stage and itr < self.stage_maxiter):			
			# To understand how to optimize each set of parameters assuming the
			# others remain constant, see math.pdf section 3.
			
			# find the f_j
			beta_j_w = self._out_dim_weights*self._beta[:, j] # weighted beta
			targets = numpy.dot(R_j, beta_j_w) / (
					  numpy.inner(self._beta[:, j], beta_j_w) + 1e-9)
			# Find the function that best fits the targets against projections.
			self._f[j], self._df[j] = self._fit_2d(p_j, targets, j, itr)
			
			# find beta_j
			f = self._f[j](p_j) # Find the n x 1 vector of function outputs.
			f_w = self._example_weights*f # f weighted by examples
			self._beta[:, j] = numpy.dot(R_j.T, f_w) / (numpy.inner(f, f_w) + 1e-9)

			# find alpha_j
			if fit_weights:
				# Find the part of the Jacobians that is common to all
				J = -(self._df[j](p_j)*numpy.sqrt(self._example_weights)*X.T).T
				JTJ = numpy.dot(J.T, J)
				A = sum([self._out_dim_weights[k] * (self._beta[k, j]**2) * JTJ
					for k in range(Y.shape[1])])
				# Collect all g_jk vectors in to a convenient matrix G_j
				G_j = R_j - numpy.outer(self._f[j](p_j), self._beta[:, j].T)
				b = -sum([self._out_dim_weights[k] * self._beta[k, j] *
					numpy.dot(J.T, G_j[:, k]) for k in range(Y.shape[1])])

				delta = numpy.linalg.lstsq(A, b, rcond=-1)[0]
				# TODO implement halving step if the loss doesn't decrease with
				# this update.
				alpha = self._alpha[:, j] + delta
				# normalize to avoid numerical drift
				self._alpha[:, j] = alpha/numpy.linalg.norm(alpha)

			# Recalculate the jth projection with new f_j and alpha_j
			p_j = numpy.dot(X, self._alpha[:, j])
			
			# Calculate mean squared error for this iteration
			prev_loss = loss
			# Subtract updated contribution of the jth term to get the
			# difference between Y and ^Y, the predictions. 
			diff = R_j - numpy.outer(self._f[j](p_j), self._beta[:, j].T)
			# multiply rows of the diff by weights, square, multiply columns
			# by other weights, and sum to get the final loss
			diff_w = (diff.T * self._example_weights).T # weighted diff
			loss = numpy.sum(self._out_dim_weights*(diff_w)**2)
			itr += 1

	def _backfit(self, X, Y, j, fit_weights):
		""" Backfitting is the process of refitting all stages after a new stage
		is found. The idea is that the new stage causes the residuals for other
		stages to change, so it may be possible to do better in fewer stages by
		accounting for this new information.

		Refitting occurs for one stage at a time, cyclically around the set
		until convergence. Refitting a stage is expensive, so backfitting can be
		extremely so. Use `backfit_maxiter` to limit the number of cycles.

		Parameters
		----------
		`X` array-like of shape = (n_samples, n_features):
			The training input samples.
		`Y` array-like, shape (n_samples, n_outputs)
			The target values.
		`j` int:
			The index of this stage in the additive model.
		`fit_weights` boolean:
			Whether to refit stages' alphas or leave them unmodified.
		"""
		itr = 0
		prev_loss = -numpy.inf
		loss = numpy.inf
		while (abs(prev_loss - loss) > self.eps_backfit and itr < self.backfit_maxiter):
			for t in range(j):
				self._fit_stage(X, Y, t, fit_weights)

			prev_loss = loss
			diff = Y - self.predict(X)
			diff_w = (diff.T * self._example_weights).T # weighted diff
			loss = numpy.sum(self._out_dim_weights*(diff_w)**2)
			itr += 1

	def _fit_2d(self, x, y, j, itr):
		""" Find a function mapping from x points in R1 to y points in R1.

		Parameters
		----------
		x array-like:
			Input points.
		y array-like:
			Target points.
		j int:
			The index of the stage that requires this fit. Only used for plots.
		itr int:
			The current iteration of the alternating optimization process that
			requires this fit. Only used for plots.

		Returns
		-------
		fit, a callable requiring numerical input
		derive, a callable requiring numerical input

		"""
		if self.fit_type == 'polyfit':
			coeffs = numpy.polyfit(x, y, deg=self.degree, w=self._example_weights)
			fit = numpy.poly1d(coeffs)
			deriv = fit.deriv(m=1)
		elif self.fit_type == 'spline':
			order = numpy.argsort(x)
			# set s according to the recommendation at: stackoverflow.com/
			# questions/8719754/scipy-interpolate-univariatespline-not-
			# smoothing-regardless-of-parameters
			fit = UnivariateSpline(x[order], y[order], w=self._example_weights,
				k=self.degree, s=len(y)*y.var(), ext=0)
			deriv = fit.derivative(1)

		# Plot the projections versus the residuals in matplotlib so the user
		# can get a picture of what is happening.
		if self.show_plots and (itr % self.plot_epoch == 0):
			pyplot.scatter(x, y)
			pyplot.title('stage ' + str(j) + ' iteration ' + str(itr))
			pyplot.xlabel('projections')
			pyplot.ylabel('residuals')
			xx = numpy.linspace(min(x), max(x), 100)
			yy = fit(xx)
			pyplot.plot(xx, yy, 'g', linewidth=1)
			pyplot.show()

		return fit, deriv


class ProjectionPursuitClassifier(BaseEstimator, ClassifierMixin):
	""" Perform classification with projection pursuit.

	Parameters
	----------
	All the same as those to the constructor of ProjectPursuitRegressor, except:

	`pairwise_loss_matrix` array-like of dimension (n_classes, n_classes),
		default=None: The adjacency matrix L has entries L[c,k]=l_ck specifying
		the weight of the penalty of predicting the answer is class k when it is
		actually class c. If unspecified, all penalties are considered to have
		the same importance.
	"""
	def __init__(self, r=10, fit_type='polyfit', degree=3, opt_level='high',
				 example_weights='uniform', pairwise_loss_matrix=None,
				 eps_stage=0.0001, eps_backfit=0.01, stage_maxiter=100,
				 backfit_maxiter=10, random_state=None, show_plots=False,
				 plot_epoch=50):

		# Do parameter checking for parameters that will not be checked when
		# the inner PPR model is constructed.
		if example_weights is not 'uniform':
			try:
				example_weights = as_float_array(example_weights)
			except (ValueError, TypeError) as error:
				raise ValueError('example_weights must be uniform or array-like.')
		if pairwise_loss_matrix is not None:
			try:
				pairwise_loss_matrix = as_float_array(pairwise_loss_matrix)
			except (ValueError, TypeError) as error:
				raise ValueError('pairwise_loss_matrix must be None or array-like.')
			if numpy.any(pairwise_loss_matrix < 0):
				raise ValueError('pairwise_loss_matrix can not contain negatives.')
			elif numpy.any(numpy.diag(pairwise_loss_matrix)) != 0:
				raise ValueError('pairwise_loss_matrix[i,i] must == 0 for all i.')

		# sklearn's clone() works by calling get_params, which calls get_param_
		# names to crawl the constructor and find out which parameters are
		# necessary to reconstruct the model without its data, and then calling
		# getattr a bunch of times to find the settings of these parameters in
		# the object. So if you simply use the parameters and don't actually
		# save them as attributes, clone will pass a bunch of Nones rather than
		# the defaults to the constructor.
		params = locals()
		for k, v in params.items():
			if k != 'self':
				setattr(self, k, v)

	def fit(self, X, Y):
		""" Train the model.

		Parameters
		----------
		`X` array-like of shape = (n_samples, n_features):
			The training input samples.
		`Y` array-like, shape = (n_samples,) or (n_samples, n_outputs)
			The target values.

		Returns
		-------
		self ProjectionPursuitClassifier:
			A trained model.
		"""
		X, Y = check_X_y(X, Y)
		if Y.ndim == 1:
			Y = Y.reshape(-1, 1) # Need 2D Y for encoding purposes

		if self.example_weights is not 'uniform' and \
			self.example_weights.shape[0] != Y.shape[0]:
			raise ValueError('If weighting examples, then n_examples needs ' + \
				'to match the length of example_weights from construction.')

		# classes_ property is required for sklearn classifiers. unique_labels
		self.classes_ = unique_labels(Y) # also performs some input validation.

		# Encode the input Y as a multi-column H.
		# TODO: CategoricalEncoder coming in sklearn v0.20 can simplify this.
		if Y.dtype.char in ['S', 'U', 'O']: # special handling for string types
			self._labeler = LabelEncoder()
			Y = self._labeler.fit_transform(Y[:,0]).reshape(-1, 1)
		else:
			self._labeler = None
		self._encoder = OneHotEncoder() # can take numerical input
		H = self._encoder.fit_transform(Y).A # .A gets the full numpy array

		# Calculate the weights. See section 4 of math.pdf.
		if self.example_weights is not 'uniform':
			pi_c = numpy.sum(H, axis=0) # pi_c * n, technically, for all c
			s_c = numpy.dot(self.example_weights, H) # find s_c for all c
			w_ex = s_c / (pi_c + 1e-9) # column weights due to example weights
		else:
			w_ex = numpy.ones(H.shape[1])

		if self.pairwise_loss_matrix is not None:
			w_pl = numpy.sum(self.pairwise_loss_matrix, axis=1) # sum over k
		else:
			w_pl = numpy.ones(H.shape[1])

		# The problem is reduced to regression
		self._ppr = ProjectionPursuitRegressor(self.r, self.fit_type,
			self.degree, self.opt_level, self.example_weights, w_ex*w_pl,
			self.eps_stage, self.eps_backfit, self.stage_maxiter,
			self.backfit_maxiter, self.random_state, self.show_plots,
			self.plot_epoch)

		self._ppr.fit(X, H)
		return self

	def predict(self, X):
		""" Use the fitted estimator to make predictions on new data.

		Parameters
		----------
		`X` array-like of shape = (n_samples, n_features):
			The input samples.

		Returns
		-------
		`Y` array of shape = (n_samples):
			The result of passing X through the evaluation function, taking
			the argmax of the output, and mapping it back to a class.
		"""
		check_is_fitted(self, '_ppr')
		H = self._ppr.predict(X)
		if H.ndim == 1: # sklearn expects a 1D answer from predict some times
			H = H.reshape(-1, 1) # but here need 2D

		# argmax gives the index of the most likely class. Map back to the class
		# itself with the encoder's active_features_ array.
		numerical_classes = self._encoder.active_features_[numpy.argmax(H, axis=1)]
		return self._labeler.inverse_transform(numerical_classes) if \
			self._labeler else numerical_classes
