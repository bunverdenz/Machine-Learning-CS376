import numpy

def save(y_pred):
	numpy.savetxt("y_pred.csv", y_pred, delimiter=",")