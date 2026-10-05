"""Entrena un modelo Prophet con el dataset de pasajeros de aerolíneas y lo guarda en disco."""

import pandas as pd
from prophet import Prophet
import pickle

# Dataset público: pasajeros mensuales de aerolíneas (1949-1960).
url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'

# --- Carga y preparación de datos ---
df = pd.read_csv(url)
# Prophet exige las columnas 'ds' (fecha) y 'y' (valor a pronosticar).
df.columns = ['ds', 'y']
df['ds'] = pd.to_datetime(df['ds'])

# --- Entrenamiento ---
# Estacionalidad multiplicativa: la amplitud estacional crece con la tendencia.
model = Prophet(seasonality_mode='multiplicative')
model.fit(df)

# --- Persistencia ---
# Se usa pickle porque el JSON de Prophet falla en Windows con fechas anteriores a 1970.
with open('prophet_model.pkl', 'wb') as f:
    pickle.dump(model, f)