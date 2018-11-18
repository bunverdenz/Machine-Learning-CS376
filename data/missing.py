#A simple script for counting the percentage and number 
#of missing values in the data
import csv

file_name = "data_train.csv"

with open(file_name, "r") as csvfile:
	rd = csv.reader(csvfile)
	total = 0
	total1 = 0
	for row in rd:
		total1+=1
		missing = False
		if not row[17]:
			continue
		for x in row:
			if x=='':
				total+=1
				break

print("Total number of tuples: {}".format(total1))
print("The number of tuples, missing at least one value: {}".format(total))
print(total*100.0/total1,'%')