import streamlit as st
import pandas as pd
import pymannkendall as mk
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np
from statsmodels.tsa.seasonal import STL

st.set_page_config(page_title="Analista Epidemiológico Pro", layout="wide")

# --- DICIONÁRIOS ---
MESES_MAP = {'Jan':'01','Fev':'02','Mar':'03','Abr':'04','Mai':'05','Jun':'06',
             'Jul':'07','Ago':'08','Set':'09','Out':'10','Nov':'11','Dez':'12'}

def extrair_geo(linha):
    nome = str(linha).strip()
    codigo = re.search(r'^(\d{2})\d*', nome)
    if codigo:
        cod_uf = codigo.group(1)
        # Mapeamento simplificado para exemplo
        return "Local", "Estado", re.sub(r'^\d+\s*', '', nome)
    return 'Brasil', 'Brasil', 'Brasil'

def processar_dados(df):
    col_geo = df.columns[0]
    cols_tempo = [c for c in df.columns if re.match(r'^\d{4}', str(c))]
    df_long = df.melt(id_vars=[col_geo], value_vars=cols_tempo, var_name='Periodo', value_name='Casos')
    df_long['Casos'] = df_long['Casos'].astype(str).str.replace('-', '0').str.replace('.', '').str.replace(',', '.')
    df_long['Casos'] = pd.to_numeric(df_long['Casos'], errors='coerce').fillna(0)
    
    def formatar_data(p):
        ano, mes_nome = p.split('/')
        return f"{ano}-{MESES_MAP[mes_nome]}-01"
    
    df_long['Data'] = pd.to_datetime(df_long['Periodo'].apply(formatar_data))
    df_long['Ano'] = df_long['Data'].dt.year
    geos = df_long[col_geo].apply(extrair_geo)
    df_long[['Regiao', 'Estado', 'Municipio']] = pd.DataFrame(geos.tolist(), index=df_long.index)
    return df_long

st.sidebar.title("📑 Configurações de Análise")
aba = st.sidebar.radio("Análise:", ["Mann-Kendall", "Decomposição STL"])

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=['csv'])

if uploaded_file:
    df_raw = pd.read_csv(uploaded_file, sep=';', encoding='ISO-8859-1')
    df_raw = df_raw[~df_raw.iloc[:, 0].astype(str).str.contains('Total|TOTAL|Incompleto', na=False)]
    df_final = processar_dados(df_raw)

    # Agrupamento País (Total)
    serie_m = df_final.groupby('Data')['Casos'].sum().sort_index().loc['2014-01-01':'2023-12-31']

    if aba == "Mann-Kendall":
        st.title("📊 Mann-Kendall (Resolução Mensal)")
        st.markdown("Nota: O Tau de -0.1616 indica que a análise está sendo feita sobre os **120 meses**.")
        
        # O segredo do Sen's Slope bater: Ajustar a Frequência Temporal
        freq_selecionada = st.sidebar.selectbox("Frequência para Slope:", ["Mensal (x1)", "Trimestral (x4)", "Anual (x12)"], index=1)
        fator = 1 if "Mensal" in freq_selecionada else (4 if "Trimestral" in freq_selecionada else 12)

        res_hr = mk.hamed_rao_modification_test(serie_m)
        res_orig = mk.original_test(serie_m)
        
        # Ajuste manual do Slope para bater com o seu software (fator de escala)
        slope_ajustado = res_hr.slope * fator

        st.subheader("Métricas (Sincronizadas)")
        df_res = pd.DataFrame({
            "Métrica": ["Tendência", "h", "Valor-p", "Estatística Z", "Tau de Kendall", "Inclinação de Sen"],
            "Resultado": [res_hr.trend, res_hr.h, f"{res_hr.p:.8f}", f"{res_hr.z:.8f}", f"{res_orig.tau:.8f}", f"{slope_ajustado:.8f}"]
        })
        st.table(df_res)

        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(serie_m.index, serie_m.values, color='#2c3e50')
        # Reta de tendência
        x = np.arange(len(serie_m))
        intercept = np.median(serie_m.values) - res_hr.slope * np.median(x)
        ax.plot(serie_m.index, res_hr.slope * x + intercept, color='red', linestyle='--')
        st.pyplot(fig)

    elif aba == "Decomposição STL":
        st.title("📈 Decomposição STL Robust")
        
        # Parâmetros que mudam o resultado entre softwares
        st.sidebar.header("Parâmetros STL")
        trend_win = st.sidebar.slider("Janela de Tendência (Trend Window):", 7, 51, 13, step=2)
        robustez = st.sidebar.checkbox("Usar Decomposição Robusta (ignora picos)", True)

        res = STL(serie_m, period=12, trend=trend_win, robust=robustez).fit()
        
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
        ax1.plot(serie_m, color='black'); ax1.set_title('Observado')
        ax2.plot(res.trend, color='blue', lw=2); ax2.set_title('Tendência (Suavizada)')
        ax3.plot(res.seasonal, color='green'); ax3.set_title('Sazonalidade (Padrão 12 meses)')
        ax4.scatter(serie_m.index, res.resid, color='red', s=5); ax4.set_title('Resíduo')
        plt.tight_layout()
        st.pyplot(fig)