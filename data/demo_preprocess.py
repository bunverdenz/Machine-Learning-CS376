import csv 
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score

f = open('data_train.csv')
csv_f = csv.reader(f)

def pre_process(indexes):
	x_train = []
	y_train = []
	for row in csv_f:
		if row[0]=='contract_date':
			continue

		features = []
		for index in indexes:
			if not row[index]:
				features.append(None)
				continue
			if index==0:
				features.append(process_date(row[index]))
			elif index==18:
				features.append(process_date(row[index]))
			elif index==19:
				features.append(year_to_days(int(row[index])))
			else:
				features.append(float(row[index]))

		x_train.append(features)
		y_train.append(float((row[23])))

	x_train = handle_nones(x_train)
	return (x_train,y_train)

def year_to_days(year):
	leap = year//4
	days = leap*366
	days += (year-leap)*365
	return days

def month_to_days(month,leap):
	monthDays = [31,28,31,30,31,30,31,31,30,31,30,31]
	if leap:
		monthDays[1] = 29
	days = 0
	for i in range(month-1):
		days+=monthDays[i]
	return days

def process_date(date):
	temp = date.split("-")
	yy = int(temp[0])
	mm = int(temp[1])
	dd = int(temp[2])
	isLeap = (yy%4==0)
	days = dd
	days += month_to_days(mm,isLeap)
	days += year_to_days(yy)
	return float(days)

def handle_nones(x_train):
	# just calculates the average for that feature
	rowLength = len(x_train[0])
	sums = [0]*rowLength
	numbers = [0]*rowLength
	for x in x_train:
		for i in range(rowLength):
			if x[i]:
				sums[i]+=x[i]
				numbers[i]+=1 
	averages = []
	for i in range(rowLength):
		averages.append(sums[i]/numbers[i])
	trainLength = len(x_train)
	for i in range(trainLength):
		for j in range(rowLength):
			if not x_train[i][j]:
				x_train[i][j] = averages[j]
	return x_train


indexes = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22]
indexes = [0,10,11,12,13,14,18,19]
x_train, y_train = pre_process(indexes)

x_train = np.array(x_train)
y_train = np.array(y_train)

# Split the data into training/testing sets
diabetes_X_train = x_train[:-220000]
diabetes_X_test = x_train[-220000:]

# Split the targets into training/testing sets
diabetes_y_train = y_train[:-220000]
diabetes_y_test = y_train[-220000:]

# Create linear regression object
regr = linear_model.LinearRegression()

# Train the model using the training sets
regr.fit(diabetes_X_train, diabetes_y_train)

# Make predictions using the testing set
diabetes_y_pred = regr.predict(diabetes_X_test)


print('Score:',regr.score(diabetes_X_test,diabetes_y_test))
# The coefficients
print('Coefficients: \n', regr.coef_)
# The mean squared error
print("Mean squared error: %.2f"
      % mean_squared_error(diabetes_y_test, diabetes_y_pred))
# Explained variance score: 1 is perfect prediction
print('Variance score: %.2f' % r2_score(diabetes_y_test, diabetes_y_pred))




