import streamlit as st
import pandas as pd
from pysus.online_data import SINAN
import pymannkendall as mk
import matplotlib.pyplot as plt

st.title("Análise de Hanseníase - Tocantins (2015-2024)")

@st.cache_data
def carregar_dados():
    # Busca dados de Hanseníase de TO
    df = SINAN.download('HANS', states='TO')
    return SINAN.to_dataframe(df)

if st.sidebar.button("Rodar Análise Estatística"):
    df = carregar_dados()
    df['ano'] = pd.to_datetime(df['DT_NOTIFIC']).dt.year
    df_filtrado = df[(df['ano'] >= 2015) & (df['ano'] <= 2024)]
    serie = df_filtrado.groupby('ano').size()

    # TESTE DE MANN-KENDALL (Igual ao artigo)
    res = mk.hamed_rao_modification_test(serie)
    
    st.metric("Tendência Detectada", res.trend)
    st.write(f"P-valor: {res.p:.4f}")

    # Gráfico de Linha
    fig, ax = plt.subplots()
    serie.plot(kind='line', marker='o', color='red', ax=ax)
    st.pyplot(fig)