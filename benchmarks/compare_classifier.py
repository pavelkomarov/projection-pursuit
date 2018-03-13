# This script gives some comparison between PPC and other classifiers.

import numpy
import time
import sys
sys.path.append("..")
from skpp import ProjectionPursuitClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_wine

rfc = RandomForestClassifier()
dtc = DecisionTreeClassifier()
mlp = MLPClassifier()
knn = KNeighborsClassifier()
svc = SVC()
ppc = ProjectionPursuitClassifier()

X, Y = load_wine(return_X_y=True)

n = int(X.shape[0]*0.8)
temp = numpy.arange(n)
numpy.random.shuffle(temp)
training = temp[0:int(n*0.8)]
testing = temp[int(n*0.8):]

a = time.time()
rfc.fit(X[training], Y[training])
b = time.time()
dtc.fit(X[training], Y[training])
c = time.time()
mlp.fit(X[training], Y[training])
d = time.time()
knn.fit(X[training], Y[training])
e = time.time()
svc.fit(X[training], Y[training])
f = time.time()
ppc.fit(X[training], Y[training])
g = time.time()

Yrfc = rfc.predict(X[training])
Ydtc = dtc.predict(X[training])
Ymlp = mlp.predict(X[training])
Yknn = knn.predict(X[training])
Ysvc = svc.predict(X[training])
Yppc = ppc.predict(X[training])

print("Accuracy on training examples for RandomForestClassifier:",
	numpy.sum(Y[training] == Yrfc)/float(Yrfc.size))
print("Accuracy on training examples for DecisionTreeClassifier:",
	numpy.sum(Y[training] == Ydtc)/float(Ydtc.size))
print("Accuracy on training examples for MLPClassifier:",
	numpy.sum(Y[training] == Ymlp)/float(Ymlp.size))
print("Accuracy on training examples for KNeighborsClassifier:",
	numpy.sum(Y[training] == Yknn)/float(Yknn.size))
print("Accuracy on training examples for SVC:",
	numpy.sum(Y[training] == Ysvc)/float(Ysvc.size))
print("Accuracy on training examples for ProjectionPursuitClassifier:",
	numpy.sum(Y[training] == Yppc)/float(Yppc.size))

Yrfc = rfc.predict(X[testing])
Ydtc = dtc.predict(X[testing])
Ymlp = mlp.predict(X[testing])
Yknn = knn.predict(X[testing])
Ysvc = svc.predict(X[testing])
Yppc = ppc.predict(X[testing])

print("Accuracy on testing examples for RandomForestClassifier:",
	numpy.sum(Y[testing] == Yrfc)/float(Yrfc.size))
print("Accuracy on testing examples for DecisionTreeClassifier:",
	numpy.sum(Y[testing] == Ydtc)/float(Ydtc.size))
print("Accuracy on testing examples for MLPClassifier:",
	numpy.sum(Y[testing] == Ymlp)/float(Ymlp.size))
print("Accuracy on testing examples for KNeighborsClassifier:",
	numpy.sum(Y[testing] == Yknn)/float(Yknn.size))
print("Accuracy on testing examples for SVC:",
	numpy.sum(Y[testing] == Ysvc)/float(Ysvc.size))
print("Accuracy on testing examples for ProjectionPursuitClassifier:",
	numpy.sum(Y[testing] == Yppc)/float(Yppc.size))

print("Training time for RandomForestClassifier:", b-a)
print("Training time for DecisionTreeClassifier:", c-b)
print("Training time for MLPClassifier:", d-c)
print("Training time for KNeighborsClassifier:", e-d)
print("Training time for SVC:", f-e)
print("Training time for ProjectionPursuitClassifier:", g-f)