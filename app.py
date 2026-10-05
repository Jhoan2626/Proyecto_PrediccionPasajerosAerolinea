import streamlit as st
import pandas as pd
import pickle

with open('prophet_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.title("Proyección de Pasajeros de Aerolíneas")

st.markdown(
    """
    Esta aplicación pronostica el número de pasajeros de aerolíneas usando el modelo **Prophet**,
    entrenado con datos históricos mensuales.

    **Cómo usarla:**
    1. En el panel lateral, ajusta el control deslizante con la cantidad de meses a proyectar.
    2. Presiona el botón **Pronosticar**.
    3. Revisa la gráfica de proyección, la gráfica de componentes (tendencia y estacionalidad) y,
       si lo deseas, la tabla de datos desplegable.
    """
)

meses = st.sidebar.slider("Selecciona la cantidad de meses a proyectar a futuro:", min_value=1, max_value=60, value=12)

if st.sidebar.button("Pronosticar"):
    futuro = model.make_future_dataframe(periods=meses, freq='MS')
    prediccion = model.predict(futuro)
    
    fig = model.plot(prediccion)
    st.pyplot(fig)
    st.pyplot(model.plot_components(prediccion))

    with st.expander("Datos de la proyección"):
        st.dataframe(
            prediccion[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(meses).rename(
                columns={'ds': 'Fecha', 'yhat': 'Pronóstico', 'yhat_lower': 'Límite Inferior', 'yhat_upper': 'Límite Superior'}
            )
        )