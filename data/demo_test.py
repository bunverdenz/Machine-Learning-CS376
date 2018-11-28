from demo_train import sklearn_linear
from demo_train import sklearn_gradient_boosting
from demo_preprocess_test import pre_process
from y_to_csv import save
from demo_train import xgboost

# sklearn Linear Regression
def run_sklearn_linear_regression(indexes):
	regr = sklearn_linear(indexes)
	Y_pred = regr.predict(x_test)
	return Y_pred

# sklearn Gradient Boosting Regression
def run_sklearn_gradient_boosting(indexes):
	clf = sklearn_gradient_boosting(indexes)
	Y_pred = clf.predict(x_test)
	return Y_pred

# xgboost regressor
def run_xgboost(indexes):
	model = xgboost(indexes)
	Y_pred = model.predict(x_test)
	return Y_pred

#indexes = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
indexes = [0,8,10,11,12,13,14,15,16,18,19]
x_test = pre_process(indexes)

Y_pred = run_xgboost(indexes)
#save(Y_pred)