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
    try:
        arquivos = SINAN.download('HANS', state='TO')
        df = SINAN.to_dataframe(arquivos)
        if not df.empty:
            df.columns = [c.upper() for c in df.columns]
            return df
    except:
        return pd.DataFrame()

# --- SIDEBAR ---
st.sidebar.header("Fonte de Dados")
opcao = st.sidebar.selectbox("Como deseja obter os dados?", 
                             ("Upload de Tabela (CSV/Excel)", "DATASUS (Extração Direta)", "Dados de Exemplo"))

serie = pd.Series()

if opcao == "Upload de Tabela (CSV/Excel)":
    uploaded_file = st.sidebar.file_uploader("Escolha o ficheiro do TabNet", type=['csv', 'xlsx'])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                # O segredo está aqui: tentar ler com latin-1 (ISO-8859-1) que é o padrão do TabNet
                try:
                    df_upload = pd.read_csv(uploaded_file, sep=None, engine='python', encoding='utf-8')
                except UnicodeDecodeError:
                    # Se falhar o UTF-8, tenta o padrão brasileiro
                    uploaded_file.seek(0) # Volta ao início do arquivo
                    df_upload = pd.read_csv(uploaded_file, sep=None, engine='python', encoding='ISO-8859-1')
            else:
                df_upload = pd.read_excel(uploaded_file)
            
            st.write("### Dados carregados com sucesso!")
            st.dataframe(df_upload.head())
            
            col_ano = st.selectbox("Coluna do ANO:", df_upload.columns)
            col_valor = st.selectbox("Coluna da QUANTIDADE:", df_upload.columns)
            
            # Limpeza: remove linhas que não são números (comum em totais no fim de tabelas do TabNet)
            df_upload[col_valor] = pd.to_numeric(df_upload[col_valor], errors='coerce')
            df_upload = df_upload.dropna(subset=[col_valor])
            
            serie = df_upload.set_index(col_ano)[col_valor].sort_index()
            
        except Exception as e:
            st.error(f"Erro ao ler o arquivo: {e}")

elif opcao == "DATASUS (Extração Direta)":
    if st.sidebar.button("Conectar ao DATASUS"):
        with st.spinner("Acedendo ao SINAN..."):
            df_real = carregar_dados_sinan()
            if not df_real.empty:
                df_real['ANO'] = pd.to_datetime(df_real['DT_NOTIFIC'], errors='coerce').dt.year
                serie = df_real[(df_real['ANO'] >= 2015) & (df_real['ANO'] <= 2024)].groupby('ANO').size()
            else:
                st.error("Servidor instável. Tente a opção de Upload com arquivo CSV.")

else: # Dados de Exemplo
    anos = list(range(2015, 2025))
    casos = [1200, 1150, 1180, 1050, 980, 850, 900, 820, 780, 750]
    serie = pd.Series(casos, index=anos)

# --- ANÁLISE ---
if not serie.empty:
    st.divider()
    st.subheader("📈 Estatística de Tendência (Mann-Kendall)")
    try:
        res = mk.hamed_rao_modification_test(serie)
        c1, c2, c3 = st.columns(3)
        c1.metric("Tendência", res.trend)
        c2.metric("P-Valor", f"{res.p:.4f}")
        c3.metric("Total", int(serie.sum()))
        
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.lineplot(x=serie.index, y=serie.values, marker='o', color='darkred')
        plt.title("Evolução Temporal de Hanseníase - TO")
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Erro no cálculo: {e}")