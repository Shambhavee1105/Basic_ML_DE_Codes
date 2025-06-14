import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# Load Dataset
# Replace with your actual file path if needed
df = pd.read_csv(r'C:\Python\dec8.csv') 

# Optionally show first few rows
print("Dataset Preview:")
print(df.head())

# Preprocessing
# Remove non-numeric columns if any (like names/IDs)
df_numeric = df.select_dtypes(include=["float64", "int64"])

# Standardize the numeric features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_numeric)

# K-Means Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)
kmeans_labels = kmeans.labels_
silhouette_kmeans = silhouette_score(X_scaled, kmeans_labels)
print("Silhouette Score (KMeans):", round(silhouette_kmeans, 3))

# Hierarchical Clustering
hierarchical = AgglomerativeClustering(n_clusters=3)
hierarchical_labels = hierarchical.fit_predict(X_scaled)
silhouette_hierarchical = silhouette_score(X_scaled, hierarchical_labels)
print("Silhouette Score (Hierarchical):", round(silhouette_hierarchical, 3))

# PCA for Visualization (2D)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df_vis = pd.DataFrame()
df_vis['PCA1'] = X_pca[:, 0]
df_vis['PCA2'] = X_pca[:, 1]
df_vis['KMeans Cluster'] = kmeans_labels
df_vis['Hierarchical Cluster'] = hierarchical_labels

# Visualization
plt.figure(figsize=(12, 5))

# KMeans
plt.subplot(1, 2, 1)
sns.scatterplot(data=df_vis, x='PCA1', y='PCA2', hue='KMeans Cluster', palette='Set2')
plt.title("K-Means Clustering")

# Hierarchical
plt.subplot(1, 2, 2)
sns.scatterplot(data=df_vis, x='PCA1', y='PCA2', hue='Hierarchical Cluster', palette='Set1')
plt.title("Hierarchical Clustering")

plt.tight_layout()
plt.show()
