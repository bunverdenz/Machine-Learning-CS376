from demo_preprocess_test import pre_process

from demo_train import xgboost
from numpy import savetxt

# xgboost regressor
def run_xgboost(indexes):
	model = xgboost(indexes)
	Y_pred = model.predict(x_test)
	return Y_pred

#indexes = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
indexes = [0,8,10,11,12,13,14,15,16,18,19]
x_test = pre_process(indexes)

Y_pred = run_xgboost(indexes)
savetxt("y_pred.csv", Y_pred)