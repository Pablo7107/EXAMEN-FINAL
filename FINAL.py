import pandas as pd
from sklearn.tree import DecisionTreeRegressor

# Datos proporcionados
anos = [2019, 2020, 2021, 2022, 2023, 2024]
demanda = [770, 780, 786, 795, 805, 800]
poblacion = [71.3, 73.0, 74.6, 76.3, 77.9, 79.5]
inpc = [73.0, 74.6, 76.3, 77.9, 79.5, 81.0]  # Valores INPC interpolados (ejemplo)

# Crear un DataFrame con los datos
data = {'año': anos, 'demanda': demanda, 'poblacion': poblacion, 'inpc': inpc, 'precio': [10, 12, 11, 13, 15, 14]}
df = pd.DataFrame(data)

# Seleccionar las variables independientes y dependiente
X = df[['poblacion', 'inpc', 'precio']]
y = df['demanda']

# Crear y entrenar el modelo
model = DecisionTreeRegressor()
model.fit(X, y)

# Datos para 2025
poblacion_2024 = 79.5  # Población en 2024
tasa_crecimiento = 0.02
poblacion_2025 = poblacion_2024 * (1 + tasa_crecimiento)
inpc_2025 = 150
precio_2025 = 5  # Supuesto para el precio en 2025

# Crear un DataFrame para la predicción
X_2025 = pd.DataFrame({'poblacion': [poblacion_2025], 'inpc': [inpc_2025], 'precio': [precio_2025]})

# Predecir la demanda para 2025
demanda_prediccion = model.predict(X_2025)

print(f"Demanda pronosticada para 2025: {demanda_prediccion[0]:.2f} ton")
