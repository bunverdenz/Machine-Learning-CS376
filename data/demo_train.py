from demo_preprocess import pre_process
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import ensemble
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor
import scipy.stats as st

from sklearn.metrics import accuracy_score
import scipy.stats as st
from sklearn.model_selection import RandomizedSearchCV

def perf(actual, pred):
    total = 0
    for i in range(len(actual)):
        total += abs(actual[i] - pred[i]) / actual[i]
    return 1 - total/len(actual)

def perf1(est, X, y):
    est.fit(X, y)
    pred = est.predict(X)
    total = 0
    for i in range(len(y)):
        total += abs(y[i] - pred[i]) / y[i]
    return 1 - total/len(y)

def perf_score(model, X, y, n_folds=5):
    kf = KFold(n_folds, shuffle=True, random_state=1000).get_n_splits(X)
    return cross_val_score(model, X, y, scoring=perf1, cv = kf)

def xgboost(indexes):
	X, Y = pre_process(indexes)

	xgb = XGBRegressor(n_estimators=25, learning_rate=0.15, gamma=0, subsample=0.75,
					   colsample_bytree=1, max_depth=10)

	score = perf_score(xgb, X, Y)
	print("Accuracy: {}".format(score.mean()))
	xgb.fit(X, Y)
	return xgb


def xgboost_test(indexes):
	X, Y = pre_process(indexes)
	estimator = XGBRegressor(nthreads=-1)
	params = {
		"n_estimators": st.randint(3, 40),
		"learning_rate": st.uniform(0.05, 0.4),
		"gamma": st.uniform(0, 10),
		"subsample": st.beta(10, 1),
		"colsample_bytree": st.beta(10, 1),
		"max_depth": st.randint(3, 40),

	}
	# Random Search Training with 5 folds CV
	clf = RandomizedSearchCV(estimator, params, cv=5,
							 n_jobs=1, n_iter=100)
	clf.fit(X, Y)
	best_params = clf.best_params_
	best_score = clf.best_score_

	print("best_params", best_params)
	print("best_score", best_score)
	return best_params, best_score








