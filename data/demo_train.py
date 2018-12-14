from demo_preprocess import pre_process
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import ensemble
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
from sklearn.metrics import accuracy_score

def sklearn_linear(indexes):
	x_train, y_train = pre_process(indexes)

	fraction = int(len(x_train)*0.85)
	# Split the data into training/testing sets
	X_train = x_train[:-fraction]
	X_test = x_train[-fraction:]

	# Split the targets into training/testing sets
	Y_train = y_train[:-fraction]
	Y_test = y_train[-fraction:]

	# Create linear regression object
	regr = linear_model.LinearRegression()

	# Train the model using the training sets
	regr.fit(X_train, Y_train)

	# Make predictions using the testing set
	Y_pred = regr.predict(X_test)


	print('Score:',regr.score(X_test,Y_test))
	# The coefficients
	print('Coefficients: \n', regr.coef_)
	# The mean squared error
	print("Mean squared error: %.2f"
      % mean_squared_error(Y_test, Y_pred))
	# Explained variance score: 1 is perfect prediction
	print('Variance score: %.2f' % r2_score(Y_test, Y_pred))

	return regr


def sklearn_gradient_boosting(indexes):
	X, Y = pre_process(indexes)

	kf = KFold(n_splits=5)

	total_score = 0

	i=0
	for train_index, test_index in kf.split(X):
		X_train, X_test = X[train_index], X[test_index]
		Y_train, Y_test = Y[train_index], Y[test_index]

		clf = ensemble.GradientBoostingRegressor(n_estimators = 25, max_depth=5, min_samples_split=3,learning_rate=0.1,loss='ls')

		clf.fit(X_train,Y_train)
		score = clf.score(X_test,Y_test)
		total_score += score
		print("Accuracy of {}'th iteration: {}".format(i, score))
		i+=1


	print("Accuracy:",total_score/i)
	

	clf.fit(X,Y)
	return clf

def xgboost(indexes):
	X, Y = pre_process(indexes)
	
	kf = KFold(n_splits=10, shuffle=False)

	total_score = 0

	i=0
	for train_index, test_index in kf.split(X):
		X_train, X_test = X[train_index], X[test_index]
		Y_train, Y_test = Y[train_index], Y[test_index]

		xgb = XGBRegressor(n_estimators=25, learning_rate=0.15, gamma=0, subsample=0.75,
                           colsample_bytree=1, max_depth=10)

		xgb.fit(X_train,Y_train)

		#y_pred = xgb.predict(X_test)
		score = xgb.score(X_test,Y_test)

		total_score += score
		print("Accuracy of {}'th iteration: {}".format(i, score))
		i+=1


	print("Accuracy:",total_score/i)

	return xgb











