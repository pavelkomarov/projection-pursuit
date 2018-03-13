# This script gives some insight in to the benefits of PPR. Specifically, on a
# small dataset a NN may not have enough examples for gradient descent to run
# to completion, whereas for the cost of a little extra training time PPR can
# fully utilize few examples. As a result PPR's training error is always less
# than that for NN, and its testing error is usually less. Occasionally PPR's
# fit is irregular, and its testing error is worse; this can probably be
# remedied by adding regularization to the loss function and rederiving the
# optimization process to accommodate it.

import numpy
import time
import sys
sys.path.append("..")
from skpp import ProjectionPursuitRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.datasets import load_boston

mlp = MLPRegressor()
ppr = ProjectionPursuitRegressor()

X, Y = load_boston(return_X_y=True)

n = int(X.shape[0]*0.8)
temp = numpy.arange(n)
numpy.random.shuffle(temp)
training = temp[0:int(n*0.8)]
testing = temp[int(n*0.8):]

a = time.time()
mlp.fit(X[training], Y[training])
b = time.time()
ppr.fit(X[training], Y[training])
c = time.time()

Ymlp = mlp.predict(X[training])
Yppr = ppr.predict(X[training])

print("Average squared error per training example for MLPRegressor:",
	numpy.sum((Y[training] - Ymlp)**2)/Ymlp.size)
print("Average squared error per training example for ProjectionPursuitRegressor:",
	numpy.sum((Y[training] - Yppr)**2)/Yppr.size)

Ymlp = mlp.predict(X[testing])
Yppr = ppr.predict(X[testing])

print("Average squared error per testing example for MLPRegressor:",
	numpy.sum((Y[testing] - Ymlp)**2)/Ymlp.size)
print("Average squared error per testing example for ProjectionPursuitRegressor:",
	numpy.sum((Y[testing] - Yppr)**2)/Yppr.size)

print("Training time for MLPRegressor:", b-a)
print("Training time for ProjectionPursuitRegressor:", c-b)