import streamlit as st
import pandas as pd
import pymannkendall as mk
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Análise Inteligente Hanseníase", layout="wide")

st.title("📊 Análise de Tendência Automática (Hamed & Rao)")

# --- LÓGICA DE DETECÇÃO AUTOMÁTICA ---
def identificar_colunas(df):
    col_ano = None
    col_casos = None
    
    # 1. Tenta achar o Ano (coluna com valores entre 2000 e 2030)
    for col in df.columns:
        # Converte para numérico e limpa
        vals = pd.to_numeric(df[col], errors='coerce').dropna()
        if not vals.empty and vals.iloc[0] > 1900 and vals.iloc[0] < 2100:
            col_ano = col
            break
            
    # 2. A outra coluna numérica com valores maiores é a de casos
    for col in df.columns:
        if col != col_ano:
            vals = pd.to_numeric(df[col], errors='coerce').dropna()
            if not vals.empty:
                col_casos = col
                break
    
    return col_ano, col_casos

# --- INTERFACE ---
uploaded_file = st.file_uploader("Arraste sua tabela do TabNet aqui (CSV ou Excel)", type=['csv', 'xlsx'])

if uploaded_file:
    try:
        # Tenta ler com diferentes encodings
        try:
            df = pd.read_csv(uploaded_file, sep=None, engine='python', encoding='utf-8')
        except:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=None, engine='python', encoding='ISO-8859-1')
        
        # Limpeza de linhas de Total/Vazias
        df = df[~df.iloc[:, 0].astype(str).str.contains('Total|TOTAL|Incompleto', na=False)]
        
        # Identificação Automática
        c_ano, c_casos = identificar_colunas(df)
        
        if c_ano and c_casos:
            # Converte e limpa
            df[c_ano] = pd.to_numeric(df[c_ano], errors='coerce')
            df[c_casos] = pd.to_numeric(df[c_casos], errors='coerce')
            df = df.dropna(subset=[c_ano, c_casos])
            
            serie = df.groupby(c_ano)[c_casos].sum().sort_index()
            serie = serie[serie.index >= 2015] # Foco no seu recorte 2015-2024

            # --- ESTATÍSTICA ---
            res = mk.hamed_rao_modification_test(serie)
            
            # Layout de Resultados
            st.subheader(f"Análise de {c_casos} por {c_ano}")
            m1, m2, m3 = st.columns(3)
            m1.metric("Tendência", res.trend.upper())
            m2.metric("P-Valor", f"{res.p:.4f}")
            m3.metric("Total de Casos", int(serie.sum()))

            # --- GRÁFICO ---
            fig, ax = plt.subplots(figsize=(10, 4))
            sns.lineplot(x=serie.index, y=serie.values, marker='o', color='#d63031', linewidth=2)
            plt.title("Série Temporal Detectada Automaticamente", fontsize=12)
            plt.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            if res.p < 0.05:
                st.success(f"A tendência de **{res.trend}** é estatisticamente significativa.")
            else:
                st.info("Não há tendência clara confirmada estatisticamente.")
        else:
            st.error("Não consegui identificar as colunas de Ano e Casos automaticamente. Verifique se o arquivo está no formato padrão do TabNet.")
            
    except Exception as e:
        st.error(f"Erro ao processar: {e}")