#A simple script for counting the percentage and number 
#of missing values in the data
import csv

file_name = "data_train.csv"

with open(file_name, "r") as csvfile:
	rd = csv.reader(csvfile)
	total = [0 for i in range(24)]
	miss = [0 for i in range(24)]
	for row in rd:
		for i, it in enumerate(row):
			total[i] += 1
			if not it:
				miss[i] += 1
	for i in range(23):
		print("The col #{} percentage is: {}".format(i, miss[i] / total[i]))
		print("The col #{} number is: {}".format(i, miss[i]))
