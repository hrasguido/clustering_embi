import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt

X = pd.read_csv('clean_data.csv')

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

inertias = []
K = range(1, 11)  # Probaremos de 1 a 10 clusters

for k in K:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertias.append(kmeans.inertia_)

# Crear un DataFrame con los valores de K y SSE
sse_df = pd.DataFrame({'K': K, 'SSE': inertias})
print("Valores de SSE para cada K:")
print(sse_df)

plt.figure(figsize=(10, 6))
plt.plot(K, inertias, 'bx-')
plt.xlabel('Numero de clusters (k)')
plt.ylabel('Suma de cuadrados intracluster (SSE)')
for k, sse in zip(K, inertias):
    plt.annotate(f'SSE: {sse:.2f}', (k, sse), textcoords="offset points", xytext=(0,10), ha='center')
plt.show()

# Punto de inflexión (opcional)
kneedle = KneeLocator(K, inertias, S=1.0, curve='convex', direction='decreasing')
print(f"El número óptimo de clusters es: {kneedle.elbow}")