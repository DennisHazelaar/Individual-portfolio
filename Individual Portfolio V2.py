"""
Created on Sun Jun  4 13:50:00 2023

@author: Dennis
"""

# Importing some all the neccessary

import pandas as pd # To load files and perform tasks
import numpy as np # linear algebra
import seaborn as sns  # data visualizations
import matplotlib.pyplot as plt # data visualizations
#import plotly  #  interactive plotting library 

df = pd.read_csv("data_cleaned_2021.csv")
df = df.drop("index",axis=1) # I droped the 'index' column because I don't need it.

# Display the first few rows of the dataframe.
df.head(2)

# Lets look at the shape of the dataset

print("No. of rows in the dataset:",df.shape[0])
print("No. of columns in the dataset:",df.shape[1])

# Scanning the dataset for missing values.

df.isnull().sum()

# Looking at some overall information and statistics about the data.

df.info()

# Provide a summary of the numerical information.

df.describe()

# Lets look at how the 'Rating' column is distributed:

plt.figure(figsize=(8,5))
plt.title('\n Distribution of Rating Column (before handling -1 values)\n', size=16, color='black')
plt.xlabel('\n Rating \n', fontsize=13, color='black')
plt.ylabel('\n Density\n', fontsize=13, color='black')
plt.xticks(fontsize=13)
plt.yticks(fontsize=12)
sns.distplot(df.Rating,color="purple")
plt.show()

# Replacing the -1 values in 'Rating' column with nan value.

df["Rating"] = df["Rating"].apply(lambda x: np.nan if x==-1 else x)

# Fill missing values with the mean of the distribution.

df["Rating"] = df["Rating"].fillna(df["Rating"].mean())

# Verifying that the replace function worked properly for the 'Rating' column.

plt.figure(figsize=(8,5))
plt.title('\n Distribution of Rating Column (after handling -1 values)\n', size=16, color='black')
plt.xlabel('\n Rating \n', fontsize=13, color='black')
plt.ylabel('\n Density\n', fontsize=13, color='black')
plt.xticks(fontsize=13)
plt.yticks(fontsize=12)
sns.distplot(df["Rating"],kde=True,color="red")
plt.show()

# Looking at the 'Age' column.

plt.figure(figsize=(8,5))
a = sns.displot(df.Age,color="darkcyan")
plt.title('\n Age Column\n', size=16, color='black')
plt.xlabel('\n Age \n', fontsize=13, color='black')
plt.ylabel('\n Count\n', fontsize=13, color='black')
plt.xticks(fontsize=13)
plt.yticks(fontsize=12)
plt.show()

# To see any outliers in the 'Age' column.

plt.figure(figsize=(8,5))
sns.boxplot(df.Age,color="indianred")
plt.title('\n Age Column Box Plot\n', size=16, color='black')
plt.xlabel('\n Age \n', fontsize=13, color='black')
plt.xticks(fontsize=13)
plt.show()

# Importing the dataset and dropping columns that is not needed.
#I changed a column name from "State" to "Job Location" because this will help merging the two dataframes together.
df2 = pd.read_csv("statelatlong.csv").rename(columns={"State":"Job Location"})
#Dropping the \column that is not required.
df2 = df2.drop("City",axis=1)
#Looking at first few records.
df2.head(5)

#Merging the two datasets based on "Job Location" column.
df = df.merge(df2, on='Job Location')

df.head(2)

#Importing the GeoPandas library.
import geopandas as gpd
#Geopandas lets you load the geometry for countries worldwide into an object called GeoDataFrame.
fig, ax = plt.subplots(figsize=(10,8))
countries = gpd.read_file(gpd.datasets.get_path("naturalearth_lowres"))
ax.set_xlim([-125,-65])
ax.set_ylim([22,55])
# Since our focus is on US, we can slice the “countries” object so it shows us US only:
countries[countries["name"] == "United States of America"].plot(color="lightgrey", ax=ax)
#Now we put points on it.
df.plot(x="Longitude", y="Latitude", kind="scatter",c="red",edgecolor="black", ax=ax)
plt.xlabel("")
plt.ylabel("")
plt.tick_params(top=False, bottom=False, left=False, right=False,labelleft=False, labelbottom=False)
plt.title('\n Visualization of Jobs in Different States of US \n', size=20, color='grey');
plt.show()

# We see few markers on the map, lets see how many unique state values are there.
df["Job Location"].nunique()

#Adding labels for the states
lab=["California","Massachusetts","New York","Virginia","Illinois","Maryland","Pennsylvania","Texas","Washington","North Carolina"]

#Lets look at the top 10 states with the most number of job postings.

fig, ax = plt.subplots(nrows=1, ncols=1)
a = sns.barplot(x=df["Job Location"].value_counts().index[0:10], y = df["Job Location"].value_counts()[0:10])

#Removing top and Right borders

sns.despine(bottom = False, left = False)

# figure size in inches
import matplotlib
from matplotlib import rcParams
rcParams['figure.figsize'] = 12,5

#Putting % on the bar plot.

spots = df["Job Location"].value_counts().index[0:10]
for p in ax.patches:
    ax.text(p.get_x() + 0.1, p.get_height()+4.5, '{:.2f}%'.format((p.get_height()/742)*100))

#Beautifying the plot
plt.title('\n States with Most Number of Jobs \n', size=16, color='black')
plt.xticks(fontsize=13)
plt.yticks(fontsize=12)
plt.xlabel('\n States \n', fontsize=13, color='black')
plt.ylabel('\n Jobcount \n', fontsize=13, color='black')
patches = [matplotlib.patches.Patch(color=sns.color_palette()[i], label=j) for i,j in zip(range(0,10),lab)]
plt.legend(handles=patches, loc="upper right")
plt.show()

#Making a DF of Average Salary of top 10 states in which job postings was maximum.

g = df.groupby("Job Location")["Avg Salary(K)"].mean().sort_values(ascending=False)[0:10]
g = g.reset_index().rename(columns={"Job Location":"Job Location","Avg Salary(K)":"Average Salary"})

# Plotting the average salary per annum for different states.
lab=["California","Illinois","District of Columbia","Massachusetts","New Jersey","Michigan","Rhode island","New York","North Carolina","Maryland"]

sns.barplot(y="Job Location", x = "Average Salary",data=g)

#Beautifying the plot

plt.title('\n Average Salary for Different States \n', size=16, color='black')
plt.xticks(fontsize=13)
plt.yticks(fontsize=12)
plt.xlabel('\n Average Salary (K) \n', fontsize=13, color='black')
plt.ylabel('\n States \n', fontsize=13, color='black')
patches = [matplotlib.patches.Patch(color=sns.color_palette()[i], label=j) for i,j in zip(range(0,10),lab)]
plt.legend(handles=patches,bbox_to_anchor=(1, 1), loc=2, borderaxespad=0.)
plt.show()

#Making a DF of Average Salary of all states in which job postings was maximum.

g1 = df.groupby("Job Location")["Avg Salary(K)"].mean().sort_values(ascending=False)
g1 = g1.reset_index().rename(columns={"Job Location": "Job Location", "Avg Salary(K)": "Average Salary"})

sns.barplot(y="Job Location", x="Average Salary", data=g1)

# Beautifying the plot
plt.title('\n Average Salary for Different States \n', size=16, color='black')
plt.xticks(fontsize=13)
plt.yticks(fontsize=5)
plt.xlabel('\n Average Salary (K) \n', fontsize=13, color='black')
plt.ylabel('\n States \n', fontsize=13, color='black')
patches = [matplotlib.patches.Patch(color=sns.color_palette()[i], label=j) for i, j in zip(range(0, 10), lab)]
plt.show()

