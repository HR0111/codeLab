import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Step 1: Read the dataset
df = pd.read_csv('sales_data_sample.csv' ,encoding='Latin-1')
# Step 2: Preprocess the data (select relevant features and scale)
# Using 'QUANTITYORDERED' and 'PRICEEACH' for clustering
features = df[['QUANTITYORDERED', 'PRICEEACH']]
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Step 3: Determine the optimal number of clusters using the elbow method
inertia = []
K = range(1, 11)  # Testing from 1 to 10 clusters

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=0)
    kmeans.fit(scaled_features)
    inertia.append(kmeans.inertia_)  # Inertia: Sum of squared distances to closest cluster center

# Step 4: Plot the elbow graph
plt.figure(figsize=(8, 5))
plt.plot(K, inertia, marker='o')
plt.title('Elbow Method for Optimal k')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia')
plt.xticks(K)
plt.grid()
plt.show()

# Step 5: Apply K-Means with the chosen number of clusters (e.g., 3)
optimal_k = 3  # Adjust based on elbow plot analysis
kmeans = KMeans(n_clusters=optimal_k, random_state=0)
df['Cluster'] = kmeans.fit_predict(scaled_features)

# Display the resulting clusters
print(df[['QUANTITYORDERED', 'PRICEEACH', 'Cluster']].head())  # Show first few rows with cluster assignments
