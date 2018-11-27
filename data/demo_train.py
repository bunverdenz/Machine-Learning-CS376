from demo_preprocess import pre_process
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import ensemble

def sklearn_linear(indexes):
	x_train, y_train = pre_process(indexes)

	# Split the data into training/testing sets
	X_train = x_train[:-220000]
	X_test = x_train[-220000:]

	# Split the targets into training/testing sets
	Y_train = y_train[:-220000]
	Y_test = y_train[-220000:]

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
	x_train, y_train = pre_process(indexes)

	fraction = int(len(x_train)*0.90)

	# Split the data into training/testing sets
	X_train = x_train[:-fraction]
	X_test = x_train[-fraction:]

	# Split the targets into training/testing sets
	Y_train = y_train[:-fraction]
	Y_test = y_train[-fraction:]

	clf = ensemble.GradientBoostingRegressor(n_estimators = 25, max_depth=5, min_samples_split=3,learning_rate=0.1,loss='ls')

	clf.fit(X_train,Y_train)

	#Y_pred = clf.predict(X_test)

	print("Accuracy:",clf.score(X_test,Y_test))

	return clf




