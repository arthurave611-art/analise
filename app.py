import streamlit as st
import pandas as pd
import pymannkendall as mk
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np

st.set_page_config(page_title="CalculaAí - Bioestatística", layout="wide")

# --- MAPEAMENTO IBGE ---
MAPA_ESTADOS = {'11':'RO','12':'AC','13':'AM','14':'RR','15':'PA','16':'AP','17':'TO','21':'MA','22':'PI','23':'CE','24':'RN','25':'PB','26':'PE','27':'AL','28':'SE','29':'BA','31':'MG','32':'ES','33':'RJ','35':'SP','41':'PR','42':'SC','43':'RS','50':'MS','51':'MT','52':'GO','53':'DF'}
MAPA_REGIOES = {'1':'Norte','2':'Nordeste','3':'Sudeste','4':'Sul','5':'Centro-Oeste'}

def extrair_geo(linha):
    nome = str(linha).strip()
    codigo = re.search(r'^(\d{2})\d*', nome)
    if codigo:
        cod_uf = codigo.group(1)
        return MAPA_REGIOES.get(cod_uf[0], 'Outros'), MAPA_ESTADOS.get(cod_uf, 'Outros'), re.sub(r'^\d+\s*', '', nome)
    return 'Brasil', 'Brasil', 'Brasil'

def processar_tabela(df):
    col_geo = df.columns[0]
    cols_tempo = [c for c in df.columns if re.match(r'^\d{4}', str(c))]
    df_long = df.melt(id_vars=[col_geo], value_vars=cols_tempo, var_name='Periodo', value_name='Casos')
    df_long['Casos'] = df_long['Casos'].astype(str).str.replace('-', '0').str.replace('.', '').str.replace(',', '.')
    df_long['Casos'] = pd.to_numeric(df_long['Casos'], errors='coerce').fillna(0)
    df_long['Ano'] = df_long['Periodo'].astype(str).str[:4].astype(int)
    geos = df_long[col_geo].apply(extrair_geo)
    df_long[['Regiao', 'Estado', 'Municipio']] = pd.DataFrame(geos.tolist(), index=df_long.index)
    return df_long

st.title("📊 Análise de Tendência: Mann-Kendall (Hamed & Rao)")
st.markdown("Fiel aos parâmetros da **Aula 4 - Pesquisa Aí**.")

file = st.file_uploader("Upload do CSV (Dengue 2014-2023)", type=['csv'])

if file:
    try:
        df_raw = pd.read_csv(file, sep=';', encoding='ISO-8859-1')
        df_raw = df_raw[~df_raw.iloc[:, 0].astype(str).str.contains('Total|TOTAL|Incompleto', na=False)]
        df_final = processar_tabela(df_raw)
        
        # --- CONFIGURAÇÕES LATERAIS ---
        st.sidebar.header("🗺️ Filtros e Estilo")
        nivel = st.sidebar.radio("Nível Geográfico:", ("País", "Região", "Estado", "Município"))
        
        if nivel == "País":
            df_temp = df_final; local = "Brasil"
        elif nivel == "Região":
            r = st.sidebar.selectbox("Região:", sorted(df_final['Regiao'].unique()))
            df_temp = df_final[df_final['Regiao'] == r]; local = r
        elif nivel == "Estado":
            e = st.sidebar.selectbox("Estado:", sorted(df_final['Estado'].unique()))
            df_temp = df_final[df_final['Estado'] == e]; local = e
        else:
            uf = st.sidebar.selectbox("UF:", sorted(df_final['Estado'].unique()))
            mun = st.sidebar.selectbox("Município:", sorted(df_final[df_final['Estado'] == uf]['Municipio'].unique()))
            df_temp = df_final[df_final['Municipio'] == mun]; local = mun

        # AJUSTE DE CORES
        c_ponto = st.sidebar.color_picker("Cor dos Pontos", "#2c3e50")
        c_trend = st.sidebar.color_picker("Cor da Tendência", "#e74c3c")

        # IMPORTANTE: Recorte Temporal para bater com seu software (Ex: 2014 em diante)
        anos_disponiveis = sorted(df_final['Ano'].unique())
        periodo = st.sidebar.select_slider("Período de Análise:", options=anos_disponiveis, value=(2014, 2023))
        
        serie = df_temp.groupby('Ano')['Casos'].sum().sort_index()
        serie = serie.loc[periodo[0]:periodo[1]]

        if len(serie) >= 4:
            # Cálculo Hamed & Rao e Original (para o Tau)
            res_hr = mk.hamed_rao_modification_test(serie)
            res_orig = mk.original_test(serie)
            
            # Tabela de Resultados (Engenharia Reversa de Atributos)
            st.subheader(f"Métricas do Teste - {local}")
            df_res = pd.DataFrame({
                "Métrica": ["Tendência", "h", "Valor-p", "Estatística Z", "Tau de Kendall", "Inclinação de Sen"],
                "Resultado": [
                    res_hr.trend, 
                    str(res_hr.h), 
                    f"{res_hr.p:.8f}", 
                    f"{res_hr.z:.8f}", 
                    f"{getattr(res_orig, 'tau', 0):.8f}", 
                    f"{res_hr.slope:.8f}"
                ]
            })
            st.table(df_res)

            # --- GRÁFICO ---
            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(serie.index, serie.values, color=c_ponto, marker='o', label='Casos Reais', linewidth=1.5)
            
            # Reta de Tendência
            x = np.arange(len(serie))
            intercept = np.median(serie.values) - res_hr.slope * np.median(x)
            ax.plot(serie.index, res_hr.slope * x + intercept, color=c_trend, linestyle='--', label='Tendência (Sen Slope)', linewidth=2)
            
            plt.title(f"Série Temporal de Dengue: {local} ({periodo[0]}-{periodo[1]})")
            plt.xticks(serie.index)
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.legend()
            st.pyplot(fig)
        else:
            st.warning("Selecione um período com pelo menos 4 anos.")
            
    except Exception as e:
        st.error(f"Erro: {e}")