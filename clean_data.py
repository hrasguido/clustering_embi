import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt



df = pd.read_excel('original_dataset_copy.xlsx')

# informacion inicial del dataframe:

#print(df.info())

# tipos del dataframe

#print(df.dtypes)

# Visualización de datos faltantes

#print(df.describe(include='all'))

# manejo de datos faltantes (imputacion)

df_cleaned = df.dropna()
#print(df_cleaned)

# identificacion de valores atipicos

# df_box = df_cleaned.drop('Fecha', axis=1)
# df_box = df_box.drop('Global', axis=1)
# df_box = df_box.drop('LATINO', axis=1)
# df_box = df_box.drop('RD-LATINO', axis=1)

# print(df_box.head)

# plt.figure(figsize=(12, 6))
# df_box.boxplot()

# plt.ylabel('EMBI+')
# plt.xticks(rotation=45)

# plt.tight_layout()
# plt.show()

# Reduccion de la dimensionalidad
df_cleaned = df_cleaned.drop('Fecha', axis=1)
df_cleaned = df_cleaned.drop('Global', axis=1)
df_cleaned = df_cleaned.drop('LATINO', axis=1)
df_cleaned = df_cleaned.drop('RD-LATINO', axis=1)
df_cleaned = df_cleaned.drop('Venezuela', axis=1)

#print(df_cleaned.info())

# Transformacion de datos

df_cleaned = df_cleaned.astype('float64')

#print(df_cleaned.info())

# Metodo del codo
## begin

# X = df_cleaned

# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X)

# inertias = []
# K = range(1, 11)  # Probaremos de 1 a 10 clusters

# for k in K:
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     kmeans.fit(X_scaled)
#     inertias.append(kmeans.inertia_)

# plt.figure(figsize=(10, 6))
# plt.plot(K, inertias, 'bx-')
# plt.xlabel('Numero de clusters (k)')
# plt.ylabel('Suma de cuadrados intracluster')
# plt.show()

# # punto de inflexión (opcional)

# kneedle = KneeLocator(K, inertias, S=1.0, curve='convex', direction='decreasing')
# print(f"El número óptimo de clusters es: {kneedle.elbow}")

##end

df_cleaned.to_csv('clean_data.csv', index=False)