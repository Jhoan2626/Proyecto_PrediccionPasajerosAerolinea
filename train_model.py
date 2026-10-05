import pandas as pd
from prophet import Prophet
import pickle

url = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/airline-passengers.csv'
df = pd.read_csv(url)
df.columns = ['ds', 'y']
df['ds'] = pd.to_datetime(df['ds'])

model = Prophet(seasonality_mode='multiplicative')
model.fit(df)

with open('prophet_model.pkl', 'wb') as f:
    pickle.dump(model, f)