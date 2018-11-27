from demo_train import sklearn_linear
from demo_preprocess_test import pre_process
from y_to_csv import save

#indexes = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
indexes = [0,10,11,12,13,14,18,19]

regr = sklearn_linear(indexes)
x_test = pre_process(indexes)
Y_pred = regr.predict(x_test)
save(Y_pred)