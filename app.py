import streamlit as st
import pandas as pd
from pysus.online_data import SINAN
import pymannkendall as mk
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Monitoramento Hanseníase TO", layout="wide")

st.title("📊 Dashboard Epidemiológico: Hanseníase em Tocantins")
st.markdown("Análise de tendência temporal baseada na metodologia de **Hamed e Rao**.")

# --- FUNÇÃO DE EXTRAÇÃO ---
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

# --- BARRA LATERAL ---
st.sidebar.header("⚙️ Configurações de Dados")
opcao = st.sidebar.selectbox("Fonte dos Dados:", 
                             ("Upload de Tabela (CSV/Excel)", "DATASUS (Extração Direta)", "Dados de Exemplo"))

serie = pd.Series()

if opcao == "Upload de Tabela (CSV/Excel)":
    uploaded_file = st.sidebar.file_uploader("Arraste o arquivo do TabNet aqui", type=['csv', 'xlsx'])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                try:
                    df_upload = pd.read_csv(uploaded_file, sep=None, engine='python', encoding='utf-8')
                except UnicodeDecodeError:
                    uploaded_file.seek(0)
                    df_upload = pd.read_csv(uploaded_file, sep=None, engine='python', encoding='ISO-8859-1')
            else:
                df_upload = pd.read_excel(uploaded_file)
            
            # Remove a linha de "Total" se existir (comum no TabNet)
            df_upload = df_upload[~df_upload.iloc[:, 0].astype(str).str.contains('Total|TOTAL|Incompleto', na=False)]

            st.write("### 📝 Ajuste de Colunas")
            col_ano = st.selectbox("Qual coluna representa o ANO?", df_upload.columns)
            col_valor = st.selectbox("Qual coluna tem a QUANTIDADE de casos?", df_upload.columns)
            
            # Tratamento dos dados para garantir que são números
            df_upload[col_ano] = pd.to_numeric(df_upload[col_ano], errors='coerce')
            df_upload[col_valor] = pd.to_numeric(df_upload[col_valor], errors='coerce')
            df_upload = df_upload.dropna(subset=[col_ano, col_valor])
            
            # Criar a série ordenada por ano
            serie = df_upload.groupby(col_ano)[col_valor].sum().sort_index()
            
        except Exception as e:
            st.error(f"Erro ao processar tabela: {e}")

elif opcao == "DATASUS (Extração Direta)":
    if st.sidebar.button("📡 Iniciar Extração"):
        with st.spinner("Conectando ao SINAN..."):
            df_real = carregar_dados_sinan()
            if not df_real.empty:
                df_real['ANO'] = pd.to_datetime(df_real['DT_NOTIFIC'], errors='coerce').dt.year
                serie = df_real[(df_real['ANO'] >= 2015) & (df_real['ANO'] <= 2024)].groupby('ANO').size()
            else:
                st.error("Servidor DATASUS ocupado. Tente o modo de Upload.")

else: # Dados de Exemplo
    serie = pd.Series({2015: 1200, 2016: 1150, 2017: 1180, 2018: 1050, 2019: 980, 
                       2020: 850, 2021: 900, 2022: 820, 2023: 780, 2024: 750})

# --- ANÁLISE E GRÁFICOS ---
if not serie.empty:
    st.divider()
    
    # Cálculos Estatísticos
    res = mk.hamed_rao_modification_test(serie)
    
    # Layout de métricas
    m1, m2, m3 = st.columns(3)
    m1.metric("Tendência (MK)", res.trend.upper())
    m2.metric("P-Valor", f"{res.p:.4f}")
    m3.metric("Total de Casos", int(serie.sum()))

    # Gráfico Principal
    st.subheader("📉 Evolução Temporal")
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.lineplot(x=serie.index, y=serie.values, marker='o', color='#d63031', linewidth=2.5, ax=ax)
    
    # Personalização para ficar acadêmico
    ax.set_title("Série Histórica de Notificações de Hanseníase - TO", fontsize=12)
    ax.set_xlabel("Ano de Notificação")
    ax.set_ylabel("Nº de Casos")
    plt.xticks(serie.index) # Garante que todos os anos apareçam
    plt.grid(True, linestyle=':', alpha=0.6)
    
    st.pyplot(fig)

    if res.p < 0.05:
        st.success(f"✅ Significância Estatística Detectada: A tendência de {res.trend} é real.")
    else:
        st.warning("⚠️ Sem Significância: As variações podem ser apenas flutuações casuais.")