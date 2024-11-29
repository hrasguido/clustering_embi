import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from kneed import KneeLocator
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.metrics import mean_squared_error


df = pd.read_csv('clean_data.csv')

X = df.iloc[:, 1:]  # Todas las columnas excepto la primera
y = df.iloc[:, 0]   # Primera columna

# Calcular la media de la variable dependiente
y_mean = np.mean(y)

# Crear un modelo "dummy" que siempre predice la media
y_pred = np.full_like(y, y_mean)

# Calcular el SSE usando mean_squared_error
sse = mean_squared_error(y, y_pred) * len(y)

print(f"La Suma de Errores Cuadráticos (SSE) es: {sse}")