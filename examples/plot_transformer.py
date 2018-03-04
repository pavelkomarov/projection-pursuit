""" Plotting the Projection Pursuit Transformation

This example creates dummy data in three dimensions and a target one-dimensional
Y, fit_transforms the X data with respect to the Y data by finding the best
alpha (and other) projection vectors such that the output can be reconstructed
and then returning X projected through those alphas, and then plots the original
and transformed X on a 3D scatter plot.

The dimension of the projected data will always be equal to r, the number of
terms in the PPR model. Here the transformation reduces the dimensionality of X
because r=2 while X lives in R^3.
"""
import numpy as np
from skpp import ProjectionPursuitRegressor
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

X = np.random.uniform(size=(50, 3))
Y = 50 - np.arange(50)

estimator = ProjectionPursuitRegressor(r=2)
X_transformed = estimator.fit_transform(X, Y)

figure = plt.figure()
axis = figure.gca(projection='3d')

axis.scatter(X[:,0], X[:,1], X[:,2], c='b', s=10, label='original data')
axis.scatter(X_transformed[:,0], X_transformed[:,1], np.zeros(50), c='r', s=10,
	label='transformed data')
axis.set_xlabel('x')
axis.set_ylabel('y')
axis.set_zlabel('z')

plt.title('Original and transformed (projected) data in 3-space.\n' +
	r'$\alpha_1 = {}^T$'.format(estimator._alpha[:,0].__repr__()[6:-1]) + '\n' +
	r'$\alpha_2 = {}^T$'.format(estimator._alpha[:,1].__repr__()[6:-1]))
plt.legend(loc=3)
plt.show()
