# Importar librerias requeridas
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as stats
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Fijar semilla de aleatoriedad, para que siempre que se ejecute contenga la misma aleatoriedad y no sea variable
SEED = 42
np.random.seed(SEED)

# Crear carpeta para guardar graficos del informe, si ya está creada no genera error, por eso el exist_ok
os.makedirs('./Reportes/Figuras', exist_ok=True)

# Carga de datos seleccionados desde la ruta donde se almacenaron
ruta_csv = './Data/Peajes.csv'
# Se lee el csv y lo almacena en la variable df
df = pd.read_csv(ruta_csv)

#Limpieza de datos
print('--- LIMPIEZA Y PREPARACIÓN DE DATOS ---')
print(f'Dimensiones iniciales del dataset: {df.shape}')

#Eliminar filas completamente duplicadas en caso de existir
filas_antes = len(df)
df = df.drop_duplicates()
print(f'Filas duplicadas eliminadas: {filas_antes - len(df)}')

# Limpiar los espacios en blanco en las columnas
columnas_texto = df.select_dtypes(include=['object']).columns
for col in columnas_texto:
  df[col] = df[col].astype(str).str.strip()

# Se convierten los datos a numericos
columnas_numericas = ['categoria_num', 'num_ejes', 'es_doble_calzada', 'tarifa']
for col in columnas_numericas:
  df[col] = pd.to_numeric(df[col], errors='coerce')

# Eliminar valores nulos
nulos_detectados = df[columnas_numericas].isnull().sum().sum()
if nulos_detectados > 0:
  df = df.dropna(subset=columnas_numericas)
print(f'Valores nulos o inconsistentes eliminados: {nulos_detectados}')

# Elegir solo valores positivos
df = df[(df['tarifa'] > 0) & (df['num_ejes'] > 0)]

print(f'Dimensiones finales del dataset limpio: {df.shape}\n')

# Analisis de datos
print('Estadisticas básicas')
# Se calculan datos estadisticos resumidos, como el promedio, valor maximo, minimo, etc
print(df.describe())

# Grafica 1:  Matriz de Correlación
plt.figure(figsize=(7, 5))
# Calcula una matriz de correlación entre las caracteristicas (variables independientes) y la variable dependiente (tarifa)
matriz_corr = df[
    ['categoria_num', 'num_ejes', 'es_doble_calzada', 'tarifa']
].corr()
# Se dibuja la gráfica
sns.heatmap(matriz_corr, annot=True, cmap='Blues', fmt='.2f', vmin=-1, vmax=1)
# Se configuran titulos, temas de figuras y se guarda la imagen
plt.title('Matriz de Correlación de Pearson')
plt.tight_layout()
plt.savefig('./Reportes/Figuras/01_matriz_correlacion.png', dpi=300)
plt.close()

# Inicio de entrenamiento
# Se almacena en x las variables independientes o caracteristicas a usar en este caso
X = df[['categoria_num', 'num_ejes', 'es_doble_calzada']]
# Se almacena en y la variable dependiente a predecir que es tarifa
y = df['tarifa']

# Dividimos la informacion, donde el 70% será para entrenamiento y el 30% para las pruebas, usando la "semilla de aleatoriedad"
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=SEED
)

# Se crea una instancia con el algoritmo que se utilizará
modelo = LinearRegression()
# Se entrena el modelo
modelo.fit(X_train, y_train)

# Se generan las predicciones luego de entrenar el modelo
# Se generan las predicciones correspondientes con los datos aprendidos
y_pred_train = modelo.predict(X_train)
# Analiza los datos nuevos que no ha usado y los predice
y_pred_test = modelo.predict(X_test)

# Se calculan los errores entre tarifa real y predicción
residuos_test = y_test - y_pred_test

# Evaluar el modelo
# Calcular el puntaje obtenido luego del entrenamiento
mae = mean_absolute_error(y_test, y_pred_test)
mse = mean_squared_error(y_test, y_pred_test)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred_test)

print('\nMÉTRICAS DE EVALUACIÓN (TEST)')
print(f'MAE: ${mae:.2f} COP')
print(f'MSE: {mse:.2f}')
print(f'RMSE: ${rmse:.2f} COP')
print(f'Puntaje final: {r2:.4f}')

# Resultados en graficas

# Gráfica 2: Valores Reales vs. Predicciones
# Para ver la linea de tendencia ideal
plt.figure(figsize=(7, 5))
sns.regplot(
    x=y_test,
    y=y_pred_test,
    scatter_kws={'color': '#1f77b4', 'alpha': 0.7},
    line_kws={'color': '#d62728', 'linewidth': 2},
)
plt.xlabel('Tarifa Real (COP)')
plt.ylabel('Tarifa Predicha (COP)')
plt.title('Valores Reales vs. Predicciones')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('./Reportes/Figuras/02_reales_vs_predichos.png', dpi=300)
plt.close()

# Gráfica 3: Residuos vs. Predicciones
# Grafica los errores que hubo en la predicción
plt.figure(figsize=(7, 5))
plt.scatter(y_pred_test, residuos_test, color='#2ca02c', alpha=0.7)
plt.axhline(y=0, color='black', linestyle='--', linewidth=1.5)
plt.xlabel('Valores Predichos (COP)')
plt.ylabel('Residuos (COP)')
plt.title('Residuos vs. Valores Predichos')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig('./Reportes/Figuras/03_residuos_vs_predicciones.png', dpi=300)
plt.close()

# Gráfica 4: Histograma
# Para comparar los residuos con la distribución normal
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sns.histplot(residuos_test, kde=True, ax=axes[0], color='#9467bd')
axes[0].set_title('Distribución de Residuos')
axes[0].set_xlabel('Residuo (COP)')

stats.probplot(residuos_test, dist='norm', plot=axes[1])
axes[1].set_title('Residuos')

plt.tight_layout()
plt.savefig('./Reportes/Figuras/04_validacion_residuos.png', dpi=300)
plt.close()

print(
    '\nGráficas guardadas con éxito en la carpeta ./Reportes/Figuras/'
)