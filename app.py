import streamlit as st
import pandas as pd
from pysus.online_data import SINAN
import pymannkendall as mk
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Análise Hanseníase TO", layout="wide")

st.title("📊 Análise de Tendência: Hanseníase em Tocantins (2015-2024)")
st.markdown("Baseado na metodologia de **Mann-Kendall (Hamed e Rao)**.")

@st.cache_data
def carregar_dados_sinan():
    """Tenta baixar os dados reais do SINAN"""
    try:
        # Tenta os formatos de argumentos mais comuns do PySUS
        arquivos = SINAN.download('HANS', state='TO')
        df = SINAN.to_dataframe(arquivos)
        if not df.empty:
            df.columns = [c.upper() for c in df.columns]
            return df
    except:
        return pd.DataFrame()

# --- SIDEBAR / MENU LATERAL ---
st.sidebar.header("Fonte de Dados")
opcao = st.sidebar.selectbox("Como deseja obter os dados?", 
                             ("Upload de Tabela (CSV/Excel)", "DATASUS (Extração Direta)", "Dados de Exemplo"))

serie = pd.Series()

if opcao == "Upload de Tabela (CSV/Excel)":
    uploaded_file = st.sidebar.file_uploader("Escolha o ficheiro extraído do TabNet", type=['csv', 'xlsx'])
    if uploaded_file is not None:
        if uploaded_file.name.endswith('.csv'):
            df_upload = pd.read_csv(uploaded_file, sep=None, engine='python')
        else:
            df_upload = pd.read_excel(uploaded_file)
        
        st.write("### Pré-visualização dos dados enviados:")
        st.dataframe(df_upload.head())
        
        # Tenta identificar colunas de Ano e Casos
        col_ano = st.selectbox("Selecione a coluna do ANO:", df_upload.columns)
        col_valor = st.selectbox("Selecione a coluna da QUANTIDADE (Casos):", df_upload.columns)
        serie = df_upload.set_index(col_ano)[col_valor].sort_index()

elif opcao == "DATASUS (Extração Direta)":
    if st.sidebar.button("Conectar ao DATASUS"):
        with st.spinner("Acedendo ao SINAN..."):
            df_real = carregar_dados_sinan()
            if not df_real.empty:
                df_real['ANO'] = pd.to_datetime(df_real['DT_NOTIFIC'], errors='coerce').dt.year
                serie = df_real[(df_real['ANO'] >= 2015) & (df_real['ANO'] <= 2024)].groupby('ANO').size()
            else:
                st.error("Servidor DATASUS instável. Use a opção de Upload.")

else: # Dados de Exemplo
    st.info("A usar dados simulados para demonstração.")
    anos = list(range(2015, 2025))
    casos = [1200, 1150, 1180, 1050, 980, 850, 900, 820, 780, 750]
    serie = pd.Series(casos, index=anos)

# --- PROCESSAMENTO ESTATÍSTICO (Mann-Kendall) ---
if not serie.empty:
    st.divider()
    st.subheader("📈 Resultados da Estatística (Hamed e Rao)")
    
    try:
        res = mk.hamed_rao_modification_test(serie)
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Tendência", res.trend)
        c2.metric("P-Valor", f"{res.p:.4f}")
        c3.metric("Total de Casos", int(serie.sum()))
        
        # Gráfico
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.lineplot(x=serie.index, y=serie.values, marker='o', color='darkred', linewidth=2)
        plt.title("Evolução Temporal de Hanseníase - TO")
        plt.grid(True, alpha=0.3)
        st.pyplot(fig)
        
        if res.p < 0.05:
            st.success("A tendência é estatisticamente significativa conforme o artigo base.")
        else:
            st.warning("Não há significância estatística na tendência observada.")
            
    except Exception as e:
        st.error(f"Erro no cálculo estatístico: {e}. Verifique se a tabela tem dados suficientes.")