# Proyecto semana 4- Regresión Lineal

Proyecto centrado en entrenar un modelo y generar predicciones por medio de la regresión lineal. Dando a conocer los resultados finales.
En este caso se elige un dataset centrado en los peajes a nivel nacional y se quiere validar como el modelo logra predecir las tarifas de acuerdo a las variables independientes planteadas.

### Variables del Modelo:
* **Variable Objetivo ($y$):** `tarifa` (Valor en pesos colombianos del peaje)
* **Predictores o variables independientes ($X$):**
  * `categoria_num`: Nivel categórico del vehículo (Categorías I a V).
  * `num_ejes`: Cantidad de ejes del vehículo (2 a 5 ejes).
  * `es_doble_calzada`: Variable binaria (1 = Doble calzada, 0 = Calzada sencilla).

## Tecnologías Utilizadas
* Python
* Pandas / Scikit-learn /NumPy /Matplotlib /Seaborn
* Github

## Estructura del Repositorio
```text
├── Data/
│   └── Peajes.csv                 # Dataset estructurado con tarifas del INVIAS
├── Reportes/
│   └── Figuras/                   # Gráficas generadas automáticamente
│       ├── 01_matriz_correlacion.png
│       ├── 02_reales_vs_predichos.png
│       ├── 03_residuos_vs_predicciones.png
│       └── 04_validacion_residuos.png
├── main.py                        # Script ejecutable principal
├── requirements.txt               # Dependencias del entorno
└── README.md                      # Documentación del proyecto
```
## Resultados del Modelo
* MAE: $1819.16 
* MSE: 4950374.42
* RMSE: $2224.94 
* Puntaje final: 0.9509

## Requisitos Previos e Instalación
* Python 3.10 o superior.


## Cómo Ejecutar el Proyecto
1.Desde visual studio code, abra la terminal y con el comando: cd "ruta donde alojará el proyecto" ingrese al folder.
2. Clonar el repositorio con el comando:
   git clone [https://github.com/tu-usuario/regresion-peajes-colombia.git](https://github.com/angiezarate2216/proyecto-peajes-colombia-ml.git)
3. Instalar dependencias usando el comando: pip install -r requirements.txt
4. Ejecutar el script principal para obtener los resultados: python main.py

## Consideraciones
1. Las figuras y gráficas ya están cargadas; pero al ejecutar el proyecto se volverán a generar en su entorno local.
2. Por consola también se darán a conocer resultados estadísticos de las variables independientes establecidas y adicional de la variable dependiente, como también se reflejarán los resultados principales una vez entrenado el modelo.
