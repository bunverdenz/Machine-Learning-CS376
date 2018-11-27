import csv 
import numpy as np

f = open('data_train.csv')
csv_f = csv.reader(f)

def pre_process(indexes):
	x_train = []
	y_train = []
	for row in csv_f:
		if row[0]=='contract_date':
			continue

		discard = False

		features = []
		for index in indexes:
			if not row[index]:
				#features.append(None)
				discard = True
				break
			if index==0:
				features.append(process_date(row[index]))
			elif index==18:
				features.append(process_date(row[index]))
			elif index==19:
				features.append(year_to_days(int(row[index])))
			else:
				features.append(float(row[index]))

		if discard:
			continue

		x_train.append(features)
		y_train.append(float((row[23])))

	#x_train = handle_nones(x_train)
	return (np.array(x_train),np.array(y_train))

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


