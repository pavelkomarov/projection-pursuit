import numpy
from skpp import MVPPR
import time

n = 1000
r = 20
degree = 3
opt_level = 'low'
fit_type = 'polyfit'
show_plots = False
plots_epoch = 50

p = 4
d = 5
rhat = 5
degreehat = 3
c = 100

X = numpy.random.rand(n, p) - 0.5
Y = numpy.zeros((n, d))
for i in range(d):
	for j in range(rhat):
		beta = numpy.random.randn(p)
		projection = X.dot(beta)
		f = numpy.poly1d(numpy.random.randint(-c, c, size=numpy.random.randint(degreehat+1)))
		Y[:, i] += f(projection)

temp = range(n)
numpy.random.shuffle(temp)
training = temp[0:int(n*0.8)]
testing = temp[int(n*0.8):]

estimator = MVPPR(r, fit_type, degree, opt_level, 0.001, 0.01)
#estimator = PPR(r, fit_type, degree, opt_level, 0.001, 0.01)

temp = numpy.zeros(d)
for i in range(d):
	num = numpy.inner(Y[training, i], Y[training, i])/len(training)
	temp[i] = num
print 'avg initial training squared error', numpy.mean(temp)

print 'training'
before = time.time()
estimator.fit(X[training, :], Y[training, :])
after = time.time()
print 'finished in', after-before, 'seconds'

predictions = estimator.predict(X[training, :])
for i in range(d):
	diff = Y[training, i] - predictions[:, i]
	num = numpy.inner(diff, diff)/len(diff)
	temp[i] = num
print 'avg final training squared error', numpy.mean(temp)

predictions = estimator.predict(X[testing, :])
for i in range(d):
	diff = Y[testing, i] - predictions[:, i]
	num = numpy.inner(diff, diff)/len(diff)
	temp[i] = num
print 'avg final testing squared error', numpy.mean(temp)
