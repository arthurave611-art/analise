import streamlit as st
import pandas as pd
import pymannkendall as mk
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
from statsmodels.tsa.seasonal import STL

st.set_page_config(page_title="Analista Epidemiológico Pro", layout="wide")

# --- DICIONÁRIOS IBGE ---
MAPA_ESTADOS = {'11':'RO','12':'AC','13':'AM','14':'RR','15':'PA','16':'AP','17':'TO','21':'MA','22':'PI','23':'CE','24':'RN','25':'PB','26':'PE','27':'AL','28':'SE','29':'BA','31':'MG','32':'ES','33':'RJ','35':'SP','41':'PR','42':'SC','43':'RS','50':'MS','51':'MT','52':'GO','53':'DF'}
MAPA_REGIOES = {'1':'Norte','2':'Nordeste','3':'Sudeste','4':'Sul','5':'Centro-Oeste'}
MESES_MAP = {'Jan':'01','Fev':'02','Mar':'03','Abr':'04','Mai':'05','Jun':'06','Jul':'07','Ago':'08','Set':'09','Out':'10','Nov':'11','Dez':'12'}

def extrair_geo(linha):
    nome = str(linha).strip()
    codigo = re.search(r'^(\d{2})\d*', nome)
    if codigo:
        cod_uf = codigo.group(1)
        return MAPA_REGIOES.get(cod_uf[0], 'Outros'), MAPA_ESTADOS.get(cod_uf, 'Outros'), re.sub(r'^\d+\s*', '', nome)
    return 'Brasil', 'Brasil', 'Brasil'

def processar_dados(df):
    col_geo = df.columns[0]
    cols_tempo = [c for c in df.columns if re.match(r'^\d{4}', str(c))]
    df_long = df.melt(id_vars=[col_geo], value_vars=cols_tempo, var_name='Periodo', value_name='Casos')
    df_long['Casos'] = df_long['Casos'].astype(str).str.replace('-', '0').str.replace('.', '').str.replace(',', '.')
    df_long['Casos'] = pd.to_numeric(df_long['Casos'], errors='coerce').fillna(0)
    df_long['Ano'] = df_long['Periodo'].astype(str).str[:4].astype(int)
    
    def formatar_data(p):
        ano, mes_nome = p.split('/')
        return f"{ano}-{MESES_MAP[mes_nome]}-01"
    
    df_long['Data'] = pd.to_datetime(df_long['Periodo'].apply(formatar_data))
    geos = df_long[col_geo].apply(extrair_geo)
    df_long[['Regiao', 'Estado', 'Municipio']] = pd.DataFrame(geos.tolist(), index=df_long.index)
    return df_long

# --- MENU DE NAVEGAÇÃO NA SIDEBAR ---
st.sidebar.title("📑 Navegação")
aba_selecionada = st.sidebar.radio("Selecione a Análise:", ["Mann-Kendall (Tendência)", "Decomposição STL (Sazonalidade)"])

uploaded_file = st.sidebar.file_uploader("Upload do arquivo CSV", type=['csv'])

if uploaded_file:
    try:
        df_raw = pd.read_csv(uploaded_file, sep=';', encoding='ISO-8859-1')
        df_raw = df_raw[~df_raw.iloc[:, 0].astype(str).str.contains('Total|TOTAL|Incompleto', na=False)]
        df_final = processar_dados(df_raw)

        # Filtros Geográficos Comuns
        st.sidebar.header("📍 Localidade")
        nivel = st.sidebar.selectbox("Nível:", ["País", "Estado", "Município"])
        
        if nivel == "País":
            df_temp = df_final; local = "Brasil"
        elif nivel == "Estado":
            est = st.sidebar.selectbox("Estado:", sorted(df_final['Estado'].unique()))
            df_temp = df_final[df_final['Estado'] == est]; local = est
        else:
            uf = st.sidebar.selectbox("UF:", sorted(df_final['Estado'].unique()))
            mun = st.sidebar.selectbox("Município:", sorted(df_final[df_final['Estado'] == uf]['Municipio'].unique()))
            df_temp = df_final[df_final['Municipio'] == mun]; local = mun

        if aba_selecionada == "Mann-Kendall (Tendência)":
            st.title("📊 Tendência de Mann-Kendall")
            serie = df_temp.groupby('Ano')['Casos'].sum().sort_index().loc[2014:2023]
            
            if len(serie) >= 4:
                res_hr = mk.hamed_rao_modification_test(serie)
                res_orig = mk.original_test(serie)
                
                st.subheader(f"Métricas - {local}")
                df_res = pd.DataFrame({
                    "Métrica": ["Tendência", "h", "Valor-p", "Estatística Z", "Tau de Kendall", "Inclinação de Sen"],
                    "Resultado": [res_hr.trend, str(res_hr.h), f"{res_hr.p:.8f}", f"{res_hr.z:.8f}", f"{res_orig.tau:.8f}", f"{res_hr.slope:.8f}"]
                })
                st.table(df_res)
                
                fig, ax = plt.subplots(figsize=(10, 4))
                ax.plot(serie.index, serie.values, marker='o', color='#2c3e50')
                x = np.arange(len(serie))
                intercept = np.median(serie.values) - res_hr.slope * np.median(x)
                ax.plot(serie.index, res_hr.slope * x + intercept, color='red', linestyle='--')
                st.pyplot(fig)

        elif aba_selecionada == "Decomposição STL (Sazonalidade)":
            st.title("📈 Decomposição STL")
            serie_mensal = df_temp.groupby('Data')['Casos'].sum().sort_index()
            serie_mensal.index.freq = 'MS'

            if len(serie_mensal) >= 24:
                res = STL(serie_mensal, period=12).fit()
                [Image of STL decomposition plots showing observed, trend, seasonal, and residual components]
                fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
                ax1.plot(serie_mensal, color='black'); ax1.set_title('Observado')
                ax2.plot(res.trend, color='blue'); ax2.set_title('Tendência')
                ax3.plot(res.1