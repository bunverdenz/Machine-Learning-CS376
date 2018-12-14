import csv 
import numpy as np
import pandas as pd
from sklearn import preprocessing

f = open('data_test.csv')
csv_f = csv.reader(f)

def pre_process(indexes):
	x_train = []
	y_train = []

	if True:
		data_df = pd.read_csv("{}data_train.csv", index_col=False,
							  skipinitialspace=True, parse_dates=[0, 18])
		data_df.index = data_df['apt_id']
		data_df.columns = ["contract_date", "latitude", "longtitude", "altitude",
						   "1st_class_reg", "2nd_class_reg", "road_id", "apt_id",
						   "floor", "angle", "area", "limit_car_are", "total_car_area",
						   "ext_car_enter", "avg_fee", "num_house", "avg_age_ppl",
						   "builder_id", "done_date", "built_year", "num_school",
						   "num_bus", "num_subway", "label"]

		# Count days since apartment was constructed --------------------------------------------------------------
		data_df["day_diff"] = (data_df["done_date"] - data_df["contract_date"]).dt.days
		data_df.drop(["contract_date"], axis=1, inplace=True)
		data_df.drop(["done_date"], axis=1, inplace=True)
		# Calculate age --------------------------------------------------------------------------------------------
		data_df["age"] = 2018 - data_df["built_year"]
		data_df.drop(["built_year"], axis=1, inplace=True)

		# Change label ---------------------------------------------------------------------------------------------
		lb = preprocessing.LabelBinarizer()
		# 1st_class_reg, try to categorize id of 1st_class into six id (see test_data)
		temp = lb.fit_transform(data_df["1st_class_reg"].values.reshape(-1, 1))
		data_df.drop(["1st_class_reg"], axis=1, inplace=True)
		data_df = pd.concat([data_df, pd.DataFrame(temp, columns=["one", "two", "three", "four", "five", "six"])],
							axis=1)

		# TODO: group lat,long for area that are close to each other -> calculate block by block area

		# TODO: avg of altitude then categorize altitude
		#     sa_df.loc[sa_df["tp"] == "DR"] = 0
		#     sa_df.loc[sa_df["tp"] == "CR"] = 1
		#     temp2 = sa_df.groupby(["ip_id"])["tp"].mean().reset_index()
		#     temp2.to_csv("{}avg_altitude.csv".format(PATH_DATA_CLEAN),index=False)
		# Save
		data_df.to_csv("{}data_df.csv", index=False)
		# TODO: check result
		# data_df.loc[["apt_id", "label"]].to_csv("{}y_train.csv".format(PATH_DATA), index=False)
		data_df.loc["label"].to_csv("{}y_train.csv", index=False)

		#   you can merge many raw data here && fill Null with 0
		data_all_df = pd.concat([data_df], axis=1, sort=False).fillna(0)
		data_all_df.to_csv("{}data_all.csv")

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

	x_train = handle_nones(x_train)
	return np.array(x_train)

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


