import numpy
from scipy.interpolate import UnivariateSpline
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.utils import as_float_array, check_random_state
from matplotlib import pyplot

class ProjectionPursuitRegressor(BaseEstimator, TransformerMixin, RegressorMixin):
	"""Projection Pursuit Regression uses a collection of vectors to multiply
	input data, thereby linearly "projecting" it in to a single dimension,
	passes these through univariate nonlinear functions, expands those outputs
	with a multiplication by output vectors (a sort of inverse projection but
	this time to the output space), and sums all the results together.

	The evaluation function for a fanciful ith example looks like:

		y_i = sum j=1 to r (f_j(x_i*alpha_j) * beta_j')

	where:
		i iterates examples, the rows of input and output matrices
		j iterates the number of terms in the PPR "additive model"
		r is the total number of projections and functions in the PPR
		y_i is a d-dimensional vector, the ith row in an output matrix Y
		x_i is a p-dimensional vector, the ith row of an input matrix X
		alpha_j is the jth projection vector in the mdoel, a p-dimensional
			vector inner-producted with x_i
		f_j is the jth function in the model, mapping from R1 -> R1
		beta_j' is the transpose of beta_j, a d-dimensional vector outer-
			producted with the result of f_j to yield a result in the output
			space
		* is a product, the inner product for x_i*alpha_j, the outer product
			for f_j * beta_j'

	This may seem very complicated, but the idea is simple:
		1. Linearly project the input down to one dimension where it is easier
			to work with, thereby sidestepping the curse of dimensionality.
		2. Find a sensible nonlinear mapping from this reduced space to
			"residuals", linear combinations of variance in the output.
		3. Unpack from the single-dimensional residual space to the output space
			with a kind of inverse projection.

	In practice a single projection-mapping-expansion is not descriptive enough
	to capture the richness of what may be a very complicated underlying
	relationship between X and Y, so it is repeated r times, each new "stage"
	only accounting for the variance left unexplained by the stages that have
	come before. Notice that, as per Taylor's Theorem and the no-doubt familiar
	universal approximation theorems, for certain classes of functions f, as r
	goes to infinity the evaluation function can approximate any continuous
	functional relationship between inputs and outputs.

	
	The (supervised) learning process consists of minimizing a standard
	quadratic cost function:

		sum i=1 to n (y_i - ^y_i)^2

	where:
		i iterates all training examples
		n is the total number of training examples
		y_i is the known answer for example i
		^y_i ("y-i-hat") is the answer predicted by the model for example i

	In words: get as close as you can for all examples. (TODO: There should
	maybe be some regularization here too.)

	Plugging the evaluation function in to the cost function yields a
	relationship between model parameters and cost or "loss". Because there
	are multiple dimensions in y_i, we introduce a sum over them so the PPR
	is motivated to make good predictions for all entries of the vector output:

		loss = sum i=1 to n (
			sum k=1 to d (
				w_k * (y_ik - sum j=1 to r (
					f_j(x_i*alpha_j) * beta_jk ))^2 ))

	where this new fauna:
		k iterates the columns of the output Y
		d is the number of outputs, the width of the output matrix Y
		w_k is a scalar weight, the relative importance of the kth output
			dimension
		y_ik is the scalar kth entry in the vector y_i, itself the ith row of Y
		beta_jk is the scalar kth entry of beta_j from the evaluation function

	The parameters we need to optimize to make the PPR "learn" are alpha_j, f_j,
	and beta_j. w_k are hyperparamters chosen by the user, just as r is chosen.


	The optimization scheme to solve for so many different paramters is
	non-obvious but straightforward:
		1. Initialize all alpha_j, f_j and beta_j to something random. Let j=1. 
		2. Find the "residual" variance undexplained by all stages fit so far.
		3. Project the input in to single dimension: X*alpha_j.
		4. Fit f_j to a weighted residual target versus projections.
		5. Use this f_j to find a better setting for beta_j.
		6. Use a Gauss-Newton scheme to solve for an update to alpha_j.
		7. Repeat steps 3-6 until f_j, beta_j, and alpha_j converge.
		8. (optional) Use the newly converged parameters to retune all previous
			f_j, beta_j, alpha_j where j < k. (backfitting)
		8. Increment j and go back to step 2 until j reaches r.

	This is a form of alternating optimization, wherein all parameters except
	one are held constant, the best setting for that parameter given those
	constants is found, and the process cycled through all parameters until
	convergence.


	Parameters
		----------
		`r` int, default=10:
			The number of terms in the underlying additive model. The input will
			be put through `r` projections, `r` functions of those projections,
			and then multiplication by `r` output vectors to determine output.

		`fit_type` {'polyfit', 'spline'}, default='polyfit':
			The kind of function to fit at each stage.

		`degree` int, default=3:
			The degree of polynomials or spline-sections used as the univariate
			approximator between projection and .

		`opt_level` {'high', 'medium', 'low'}, default='high':
			'low' opt_level will disable backfitting. 'medium' backfits
			previous 2D functional fits only (not projections). 'high' backfits
			everything.

		'weights' string or array-like, default='inverse-variance':
			The relative importances given to output dimensions when calculating
			the weighted residual (output of the univariate functions f_j). If
			all dimensions are of the same importance, but outputs are of
			different scales, then using the inverse variance is a good choice.
			Possible values:
				`'inverse-variance'`: Divide outputs by their variances.
				`'uniform'`: Use a vector of ones as the weights.
				`array`: Provide a custom vector of weights of dimension
						 (n_outputs,)

		`eps_stage` float, default=0.001:
			The mean squared difference between the predictions of the PPR at
			subsequent iterations of a "stage" (fitting an f, beta pair) must
			reach below this epsilon in order for the stage to be considered
			converged.

		`eps_backfit` float, default=0.01:
			The mean squared difference between the predictions of the PPR at
			subsequent iterations of a "backfit" must reach below this epsilon
			in order for backfitting to be considered converged.

		`stage_maxiter` int, default=100:
			If a stage does not converge within this many iterations, end the
			loop and move on. This is useful for divergent cases.

		`backfit_maxiter` int, default=10:
			If a backfit does not converge withint this many iterations, end
			the loop and move on. Smaller values may be preferred here since
			backfit iterations are expensive.

		`show_plots` boolean, default=False:
			Whether to produce plots of projections versus residual variance
			throughout the training process.

		`plot_epoch` int, default=50:
			If plots are displayed, show them every `plot_epoch` iterations
			of the stage-fitting process.
	"""
	def __init__(self, r=10, fit_type='polyfit', degree=3, opt_level='high',
				 weights='inverse-variance', eps_stage=0.0001, eps_backfit=0.01,
				 stage_maxiter=100, backfit_maxiter=10, random_state=None,
				 show_plots=False, plot_epoch=50):

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
		if weights not in ['inverse-variance', 'uniform']:
			try:
				weights = as_float_array(weights)
			except (TypeError, ValueError) as error:
				raise ValueError('weights must be either inverse-variance, ' + \
					'uniform, or array-like.')
			if numpy.any(weights < 0):
				raise ValueError('weights can not contain negative values.')
		if eps_stage <= 0 or eps_backfit <= 0:
			raise ValueError('Epsilons must be > 0.')
		if not isinstance(stage_maxiter, int) or stage_maxiter <= 0 or \
			not isinstance(backfit_maxiter, int) or backfit_maxiter <= 0:
			raise ValueError('Maximum iteration settings must be ints > 0.')

		self.r = r
		self.fit_type = fit_type
		self.degree = degree
		self.opt_level = opt_level
		self.weights = weights
		self.eps_stage = eps_stage
		self.eps_backfit = eps_backfit
		self.stage_maxiter = stage_maxiter
		self.backfit_maxiter = backfit_maxiter
		self.random_state = random_state
		self.show_plots = show_plots
		self.plot_epoch = plot_epoch

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

		# Sklearn does not allow mutation of object parameters (the ones not
		# prepended by an underscore), so construct or reassign weights to
		# _weights
		if self.weights == 'inverse-variance':
			variances = Y.var(axis=0)
			if max(variances) == 0:
				raise ValueError('Y must have some variance.')
			# There is a problem if a variance for any column is zero, because
			# its relative scale will appear infinite. Fill zeros with the max
			# of variances, so the corresponding columns have small weight and
			# are not major determiners of loss.
			variances[variances == 0] = max(variances)
			self._weights = 1./variances
		elif self.weights == 'uniform':
			self._weights = numpy.ones(Y.shape[1])
		elif isinstance(self.weights, numpy.ndarray):
			if Y.shape[1] != self.weights.shape[0]:
				raise ValueError('weights provided to the constructor have ' +
					'dimension ' + self.weights.shape[0] + ', which disagrees '
					+ 'with the width of Y:' + Y.shape[1])
			else:
				self._weights = self.weights

		self._random = check_random_state(self.random_state)

		# Now that input and output dimensions are known, parameters vectors
		# and can be initialized. Vectors are always stored vertically.
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
			Whether to fit 

		Returns
		-------
		self ProjectionPursuitRegressor:
			A trained model.
		"""
		# projections P = X*Alphas, P_j = X*alpha_j
		P = self.transform(X) # the n x r projections matrix
		# residuals matrix for the jth term R_j =
		#	Y - sum t=1 to r but =/= j ( f(P_t) * beta_t' )
		# Essentially, take the evaluation function, separate out the
		# contribution of the jth term, and algebraically solve for it by
		# subtracting the rest of the sum from both sides.
		R_j = Y - sum([numpy.outer(self._f[t](P[:, t]), self._beta[:, t].T) for t 
			in range(self.r) if t is not j]) # the n x d residuals matrix

		# main alternating optimization loop
		n = 0 # iteration counter
		# Start off with dummy infinite losses to get the loop started because
		# no value of loss should be able to accidentally fall within epsilon of
		# a dummy value and cause the loop to prematurely terminate.
		prev_loss = -numpy.inf
		loss = numpy.inf
		p_j = P[:,j] # n x 1, the jth column of the projections matrix
		# Use the absolute value between loss and previous loss because we do
		# not want to terminate if the unstability of parameters in the first
		# few iterations causes loss to momentarily increase.
		while (abs(prev_loss - loss) > self.eps_stage and n < self.stage_maxiter):			
			# To understand how to optimize each set of parameters assuming the
			# others remain constant, let
			#
			#	g = w(r - f*b)^2
			#
			# be the weighted squared term from inside the sums of the loss
			# function in simplified form.
			#
			# Use good ol' calculus to optimize for f:
			#
			#	dg/df = 2w(r-f*b)*-b = 0 -> -2wrb + 2fw(b^2) = 0
			#			-> f* = wrb/(w(b^2))
			#
			#	Remember b is actually a beta vector here, so b/(b^2) isn't
			#	really just b.
			#
			#	The full expression for the optimal mapping of the jth f* at the
			#	ith point becomes:
			#
			#		f*_j(x_i*alpha_j) = sum k=1 to d (w_k*beta_jk*r_ijk) /
			#							sum k=1 to d (w_k*beta_jk^2)
			#						  = a weighted residual target
			#
			#		where r_ijk is R_j[i, k]
			w_R_j = numpy.dot(R_j, self._weights*self._beta[:, j]) / (numpy.inner(
				self._beta[:, j], self._weights*self._beta[:, j]) + 1e-9)
			# Find the function that best fits the weighted residuals against
			# the projections.
			self._f[j], self._df[j] = self._fit_2d(p_j, w_R_j, j, n)
			
			# Got that? Now use g + calculus again to optimize for beta:
			#
			#	dg/db = 2w(r-f*b)*-f = 0 -> -2wrf + 2w(f^2)b = 0
			#			-> b* = wrf/(w(f^2))
			#
			#	Once again, f is actually a vector of output values at many
			#	points, so f/(f^2) isn't really just f.
			#
			#	The full expression for the optimal jth beta vector's kth entry
			#	becomes:
			#
			#		beta*_jk = sum i=1 to n (w_k*r_ijk*f_j(x_i*alpha_j)) /
			#				   sum i=1 to n (w_k*f_j(x_i*alpha_j))
			#
			#	Here w_k actually does cancel because there is no sum over k.
			#	Doing this and vectorizing to get rid of the sum over i yields:
			#
			#		beta*_jk = r_jk' * f_j(X*alpha_j) /
			#				   f_j(X*alpha_j)' * f_j(X*alpha_j)
			#
			#		where r_jk' is the transpose of r_jk, the kth column of R_j.
			f = self._f[j](p_j) # Find the n x 1 vector of function outputs.
			self._beta[:, j] = numpy.dot(R_j.T, f) / (numpy.inner(f, f) + 1e-9)

			# Now for the hard stuff. The alpha vector isn't like the other
			# parameters because it is inside the function f, so the approach
			# taken to optimize the last two parameter sets does not apply.
			#
			# Thankfully, if we let
			#
			#	g(a) = r - f(a)*b
			#
			# or in more detailed, properly subscripted, and vectorized form
			# (so the sum over i disapears):
			#
			#	g_jk(alpha_j) = r_jk - f_j(X*alpha_j) * beta_jk
			#
			# where g is subscripted by j for belonging to the jth stage of the
			# additive model and a k for belonging to the kth output dimension,
			# r_jk is the kth column of the residual matrix for the jth stage
			# R_j, and beta_jk is the scalar kth element of beta_j, then there
			# is a method called Gauss-Newton to optimize problems of the form
			#
			#	sum k=1 to d ( g_jk(a) )^2
			#
			# which is the same form as the loss function!
			#
			# The canonical solution is a_next = a - pinv(J)*g(a), where pinv
			# is the pseudoinverse (J'*J)^-1 * J', and J is the Jacobian matrix
			# of g. Here it's a little more complicated because there are d 
			# (n_outputs) different g_jk functions, but we do need to define
			# the Jacobian for the kth one:
			#
			#	J_k[u,v] = dg_jk[u](a) / da[v]
			#
			# That is: The entry at the (u, v)th location of J_k is the partial
			# derivative of the uth entry of g_jk with respect to the vth element
			# of a evaluated at the current a.
			#
			# Some time with a piece of paper yields:
			#
			#	g_jk(alpha_j)[u] = r_jk - f_j(x_u*alpha_j) * beta_jk
			#	-> dg_jk[u](a) / da[v] = -df_j(x_u*alpha_j) * beta_jk * x_uv
			#
			#	where x_u is the uth row of X. Basically for the derivative just
			#	drop the constant r_jk and multiply by the constant coefficient
			#	of alpha_j[v] = X[u, v]
			#
			#	J_k = [ -df_j(x_0*alpha_j)*   -df_j(x_1*alpha_j)*             ]
			#		  [       beta_jk*x_00          beta_jk*x_10      ...     ]
			#		  [                                                       ]
			#		  [ -df_j(x_0*alpha_j)*   -df_j(x_1*alpha_j)*             ]
			#		  [       beta_jk*x_01          beta_jk*x_11      ...     ]
			#		  [                                                       ]
			#		  [       ...                   ...                       ]
			#		  [                                     -df(x_n*alpha_j)* ]
			#         [                                         beta_jk*x_np  ]
			#
			#	= -beta_jk.*[            |                              |           ]
			#			    [ df_j(x_0*alpha_j).*x_0   ...   df_j(x_n*alpha_j).*x_n ]
			#				[            |                              |           ]
			#
			#	= -beta_jk.*df_j(X*alpha_j) O* X
			#
			#	where .* is the pointwise product of a matrix or vector with a
			#	scalar, and O* means the Hadamard product of the n-vector to the
			#	left with each of the p columns of the n x p matrix to the right.
			#
			# Great, so now let's update alpha. Unpacking the canonical update
			# and accounting for the weights yields the corresponding least-
			# squares problem:
			#
			#	sum k=1 to d (w_k.*J_k'*J_k) * delta = -sum k=1 to d (w_k J_k'*g_jk)
			#
			# where g_jk is the vector values of the function g_jk evaluated at
			# the current alpha_j and delta is a_next - a, the update to alpha.
			#
			# Let 	A = sum k=1 to d (w_k.*J_k'*J_k)
			#		b = -sum k=1 to d (w_k.*J_k'*g_jk)
			# and solve A*delta = b with a least-squares solver.
			if fit_weights:
				# Find the part of the Jacobians that is common to all
				J = -(self._df[j](p_j)*X.T).T
				JTJ = numpy.dot(J.T, J)
				A = sum([self._weights[k] * (self._beta[k, j]**2) * JTJ
					for k in range(Y.shape[1])])
				# Collect all g_jk vectors in to a convenient matrix G_j
				G_j = R_j - numpy.outer(self._f[j](p_j), self._beta[:, j].T)
				b = -sum([self._weights[k] * self._beta[k, j] *
					numpy.dot(J.T, G_j[:, k]) for k in range(Y.shape[1])])

				delta = numpy.linalg.lstsq(A, b)[0]
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
			# multiply columns of the diff by weights, square, and sum
			loss = numpy.sum((self._weights*diff)**2)
			n += 1

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
			Whether to fit 

		Returns
		-------
		self ProjectionPursuitRegressor:
			A trained model.
		"""
		n = 0
		prev_loss = -numpy.inf
		loss = numpy.inf
		while (abs(prev_loss - loss) > self.eps_backfit and n < self.backfit_maxiter):
			for t in range(j):
				self._fit_stage(X, Y, t, fit_weights)

			prev_loss = loss
			diff = Y - self.predict(X)			
			loss = numpy.sum((self._weights*diff)**2)
			n += 1

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

		# Plot the projections versus the residuals in matplotlib so the user
		# can get a picture of what is happening.
		if self.show_plots and (itr % self.plot_epoch == 0):
			pyplot.scatter(x, y)
			pyplot.title('plot' + str(itr) + 'stage' + str(j))
			pyplot.xlabel('projections')
			pyplot.ylabel('residuals')
			xx = numpy.linspace(min(x), max(x), 100)
			yy = fit(xx)
			pyplot.plot(xx, yy, 'g', linewidth=1)
			pyplot.show()

		return fit, deriv


class ProjectionPursuitClassifier(BaseEstimator):
	""" Perform classification with projection pursuit. Let risk R be

		R = sum i=1 to n (
			min over k in [1,q] (
				sum over c=1 to q (
					l_ck * p(c | x_i) )))

	where
		i iterates over examples
		min over k implements the optimal decision rule for each example
		l_ck is the user-specified loss for predicting y=k when in truth y=c
		the inner sum is the total loss for predicting y=k
		and p(c | x_i) is the true probability y=c given input x_i

	"""
	def __init__(self):
		pass