import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Завантаження даних
data = pd.read_csv("futuristic_city_traffic.csv").head(7000)

# Вибір ознак для кластеризації
features = data[['Hour Of Day', 'Traffic Density']]

# Масштабування ознак
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Кількість кластерів
k = 3

# Виконання кластеризації за допомогою методу К-середніх
kmeans = KMeans(n_clusters=k, random_state=42)
kmeans.fit(scaled_features)

# Додавання міток кластерів до даних
data['Cluster'] = kmeans.labels_

# Виведення результатів кластеризації
print("City - Cluster")
unique_cities = data[['City', 'Cluster']].drop_duplicates()
print(unique_cities)

print("Raw - Cluster")
for city, cluster in zip(data['City'], data['Cluster']):
    print(city, "-", cluster)

# Візуалізація результатів кластеризації
plt.scatter(data['Hour Of Day'], data['Traffic Density'], c=data['Cluster'], cmap='viridis')
plt.xlabel('Hour Of Day')
plt.ylabel('Traffic Density')
plt.title('Clustering of City Traffic Density')
plt.show()
