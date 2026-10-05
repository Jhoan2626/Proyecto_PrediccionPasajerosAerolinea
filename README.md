# Proyección de Pasajeros de Aerolíneas - Modelo Prophet

## Descripción del Proyecto
Implementación de un modelo de Data Analytics de series de tiempo utilizando Prophet[cite: 1]. El proyecto pronostica el tráfico de pasajeros aéreos a partir del dataset histórico AirPassengers[cite: 1] y muestra los resultados a través de una interfaz interactiva[cite: 1].

## Archivos Principales
* **`train_model.py`:** Script para la preparación de datos, entrenamiento del algoritmo y exportación del modelo[cite: 1].
* **`app.py`:** Aplicación web construida con Streamlit que permite seleccionar los meses a proyectar y visualizar gráficos de resultados[cite: 1].
* **`prophet_model.pkl`:** Archivo binario con el modelo previamente entrenado.

## Instrucciones de Ejecución Local
1. Instalar las dependencias necesarias:
`pip install pandas prophet streamlit matplotlib`

2. Entrenar el modelo (si se requiere actualizar):
`python train_model.py`

3. Ejecutar la aplicación web:
`streamlit run app.py`
