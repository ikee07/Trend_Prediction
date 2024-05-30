
import datetime
import pandas as pd
import numpy as np
import statistics as stats
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline


# Read the CSV file for the data of 6 months
data = pd.read_csv("Covid19_6months.csv",  low_memory=False)
data.head()
to_drop = ['Total_cases', 'New_cases'] # remove 'Date' since it is not used
data.drop(to_drop, inplace=True, axis=1)
print("---- 6 Months ----")
print(data.head())
print()

# Read The CSV File for the full 12months
data_all = pd.read_csv("COVID19_12months.csv",  low_memory=False)
data_all.head()
to_drop_all = ['Total_cases', 'New_cases'] # remove 'Date' since it is not used
data_all.drop(to_drop_all, inplace=True, axis=1)
print("---- 12 Months ----")
print(data_all.head())
print()

# Arrays to hold the information in the CSV Files
date = []
cases = []
all_date = []
all_cases = []


# Open File to split the data to store into the arrays
with open("Covid19_6months.csv","r") as f:
    f.readline()
    for l in f:
        l.strip()
        things=l.split(",")
        if things[0]:
            date.append([datetime.datetime.strptime(things[0],"%m/%d/%Y").timestamp()])
            cases.append([float(things[2])])
            
            all_date.append([datetime.datetime.strptime(things[0],"%m/%d/%Y").timestamp()])
            all_cases.append([float(things[2])])
    f.close()

# Open File to split the data to store into the arrays
with open("COVID19_12months.csv","r") as f:
    f.readline()
    for l in f:
        l.strip()
        things=l.split(",")
        if things[0]:
            all_date.append([datetime.datetime.strptime(things[0],"%m/%d/%Y").timestamp()])
            all_cases.append([float(things[2])])
    f.close()

# Plotting both the Prediction and Actual Graphs
model = make_pipeline(PolynomialFeatures(3), Ridge())
model.fit(date, cases)
y_plot = model.predict(date)
all_y_plot = model.predict(all_date)

# Regression Graph of 6 months
plt.title('Covid Cases 01/10/2020-06/30/2020')
plt.plot(date, cases, "b")
plt.plot(date, y_plot, "g")
plt.plot(all_date, all_cases, "r")
plt.legend(["Data Used For Prediction", "Ridge Regression Prediction", "All Data"])
plt.show()

# Comparison Graph of 12 Months
plt.title('Covid Cases 01/10/2020-1/30/2021')
plt.plot(all_date, all_cases, "g")
plt.plot(date, cases)
plt.plot(all_date, all_y_plot, "r")
plt.legend(["Data Used For Prediction", "All Data", "Ridge Regression Prediction"])
plt.show()
