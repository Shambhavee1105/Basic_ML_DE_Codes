import numpy as np
import pandas as pd

#1)Loading the dataset
file_path = r'C:/Users/Shambhavee Gune/Downloads/hd/newdata.csv'

#2)Reading the csv file
df = pd.read_csv(file_path)

#3)Displaying the original dataset
print("Original dataset: ")
print(df)

#4)Identify the missing values
print("\nMissing Values Summary:")
print(df.isnull().sum())

#5)Ignoring the missing values

# Drop rows with any NaN values
df_drop_any = df.dropna()
print("\nDataset after dropping rows with any NaN values:")
print(df_drop_any)

# Drop rows where the entire row is NaN
df_drop_all = df.dropna(how='all')
print("\nDataset after dropping rows where all values are NaN:")
print(df_drop_all)

# Drop rows with more than 2 NaN values
df_drop_thresh = df.dropna(thresh=len(df.columns) - 2)
print("\nDataset after dropping rows with more than 2 NaN values:")
print(df_drop_thresh)

# Drop NaN values in a specific column (e.g., 'Column1')
column_to_clean = df.columns[0] # Replace with the actual column name
df_drop_specific = df.dropna(subset=[column_to_clean])
print(f"\nDataset after dropping rows with NaN in {column_to_clean}:")
print(df_drop_specific)


#6)using default values to handle missing data
# Fill NaN values with 0
df_default = df.fillna(0)
print("\nDataset after replacing NaN with default value (0):")
print(df_default)

# 7) Impute values using mean, median, etc.
# Impute missing values in a specific column (e.g., 'Column2') with mean
numeric_columns = df.select_dtypes(include=[np.number]).columns

if not numeric_columns.empty:
    column_to_impute = 'Column1'  # Replace with the actual column name

    print(f"Missing values in {column_to_impute}: {df[column_to_impute].isnull().sum()}")

    df_impute_mean = df.copy()
    df_impute_mean[column_to_impute] = df_impute_mean[column_to_impute].fillna(df_impute_mean[column_to_impute].mean())

    print(f"\nDataset after imputing missing values in {column_to_impute} with mean:")
    print(df_impute_mean)

else:
    print("\nNo numeric columns found for imputation.")


# 8) Identify duplicates
duplicates = df.duplicated()
print("\nDuplicate rows in the dataset:")
print(df[duplicates])

# 9) Remove the duplicates
df_no_duplicates = df.drop_duplicates()
print("\nDataset after removing duplicates:")
print(df_no_duplicates)

# 10) Handle data redundancy
# Identify columns with identical data
redundant_columns = [col for col in df.columns if df[col].nunique() == 1]
print("\nColumns with redundant data:")
print(redundant_columns)

# Drop redundant columns (if any)
df_no_redundancy = df.drop(columns=redundant_columns)
print("\nDataset after handling redundancy:")
print(df_no_redundancy)