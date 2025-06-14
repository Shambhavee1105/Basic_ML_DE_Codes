
#Importing Libraries
import pandas as pd
import numpy as nm

#Importing Dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"

#Renaming coloumn names
columns = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'species']

#Load data into dataframe
df = pd.read_csv(url, header=None, names=columns)

#Displaying information from dataset
print("DataFrame Information:\n")
print(df.info())

#Displaying first 5 rows 
print("First 5 rows of the DataFrame:\n")
print(df.head())

#Displaying data description
print("\nDataFrame description:\n")
print(df.describe())

#Adding new coloumn
df['sepal_area'] = df['sepal_length'] * df['sepal_width']
print("First 5 rows with sepal area:\n")
print(df.head())

#Performing basic statistical opeartions
mean_sepal_length = df['sepal_length'].mean()
median_sepal_length = df['sepal_length'].median()
mode_species = df['species'].mode()[0]
std_sepal_length = df['sepal_length'].std()

#Displaying the output of statistical operations
print("\nStatistical Operations:\n")
print(f"Mean of Sepal Length: {mean_sepal_length}")
print(f"Median of Sepal Length: {median_sepal_length}")
print(f"Mode of Species: {mode_species}")
print(f"Standard Deviation of Sepal Length: {std_sepal_length}")
