import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.decomposition import PCA
from sklearn.datasets import load_iris

# Loading the Iris dataset
iris = load_iris()
columns = iris.feature_names

df = pd.DataFrame(data=iris.data, columns=columns)
df['target'] = iris.target

print("Original dataset: ")
print(df.head())

# 1) Identification of missing values
print("\nMissing values in dataset:")
print(df.isnull().sum())

# 2) Identification of duplicates
duplicates = df.duplicated()
print(f"\nNumber of duplicate rows: {duplicates.sum()}")

# 3) Handling data redundancy
df = df.drop_duplicates()
print(f"\nDataset after removing duplicates: {df.shape}")

# 4) Perform correlation analysis (Pearson's ratio)
correlation_matrix = df.select_dtypes(include=[np.number]).corr(method='pearson')
print("\nCorrelation matrix: ")
print(correlation_matrix)

# 5) Display the Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Heatmap of Correlation Matrix')
plt.show()

# 6) Data visualization 
df.hist(figsize=(12,10))
plt.suptitle('Histograms of dataset features')
plt.show()

if df.shape[1] >= 2:
    plt.scatter(df.iloc[:, 0], df.iloc[:, 1])
    plt.title('Scatter Plot between First Two Features')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.show()

# 7) Performing Min-Max scaling
scaler = MinMaxScaler()
scaled_minmax = scaler.fit_transform(df.select_dtypes(include=[np.number]))
print("\nMin-Max Scaled Data:")
print(pd.DataFrame(scaled_minmax, columns=df.select_dtypes(include=[np.number]).columns).head())

# 8) Performing Z-score scaling
scaler = StandardScaler()
scaled_zscore = scaler.fit_transform(df.select_dtypes(include=[np.number]))
print("\nZ-scored Scaled Data:")
print(pd.DataFrame(scaled_zscore, columns=df.select_dtypes(include=[np.number]).columns).head())

# 9) Data smoothing using binning method
bin_column = df.select_dtypes(include=[np.number]).columns[0]
bins = np.linspace(df[bin_column].min(), df[bin_column].max(), 4)
bin_labels = ['low', 'mid', 'high']
df['Binned_' + bin_column] = pd.cut(df[bin_column], bins=bins, labels=bin_labels, include_lowest=True)
print("\nDataset after binning:")
print(df[['Binned_' + bin_column]].head())

# 10) Feature reduction with selection to select most significant features
if df.shape[1] > 1:
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    selector = SelectKBest(score_func=f_classif, k=2)
    X_new = selector.fit_transform(X.select_dtypes(include=[np.number]), y)
    print("\nFeature Scores:")
    print(selector.scores_)

# PCA
pca = PCA(n_components=2)
reduced_data = pca.fit_transform(df.select_dtypes(include=[np.number]))
print("\nPCA Reduced Data:")
print(reduced_data[:5])