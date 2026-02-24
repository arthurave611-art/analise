import streamlit as st
import pandas as pd
from pysus.online_data import SINAN
import pymannkendall as mk
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Hanseníase TO - Análise", layout="wide")

st.title("📊 Análise de Tendência: Hanseníase em Tocantins (2015-2024)")
st.markdown("Baseado na metodologia de **Mann-Kendall (Hamed e Rao)**.")

@st.cache_data
def carregar_dados_sinan():
    """Tenta baixar os dados usando diferentes padrões de versão do PySUS"""
    # Lista de tentativas de argumentos para diferentes versões da biblioteca
    tentativas = [
        lambda: SINAN.download('HANS', state='TO'),
        lambda: SINAN.download('HANS', states='TO'),
        lambda: SINAN.download('HANS', 'TO'),
        lambda: SINAN.download('HANS', ['TO'])
    ]
    
    for tentativa in tentativas:
        try:
            arquivos = tentativa()
            df = SINAN.to_dataframe(arquivos)
            if not df.empty:
                df.columns = [c.upper() for c in df.columns]
                return df
        except Exception:
            continue
            
    return pd.DataFrame()

# Menu lateral
st.sidebar.header("Configurações")
if st.sidebar.button("Extrair e Analisar"):
    with st.spinner("Conectando ao DATASUS (SINAN)..."):
        df = carregar_dados_sinan()
        
        if not df.empty:
            # Identifica a coluna de data (DT_NOTIFIC ou similar)
            col_data = 'DT_NOTIFIC' if 'DT_NOTIFIC' in df.columns else 'DT_NOTIFICACAO'
            
            if col_data in df.columns:
                df[col_data] = pd.to_datetime(df[col_data], errors='coerce')
                df['ANO'] = df[col_data].dt.year
                
                # Filtro 2015-2024
                df_filtrado = df[(df['ANO'] >= 2015) & (df['ANO'] <= 2024)]
                serie = df_filtrado.groupby('ANO').size()
                
                if len(serie) > 1:
                    st.subheader("📈 Estatísticas de Tendência")
                    res = mk.hamed_rao_modification_test(serie)
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Tendência", res.trend)
                    c2.metric("P-Valor", f"{res.p:.4f}")
                    c3.metric("Total de Casos", len(df_filtrado))
                    
                    # Gráfico
                    fig, ax = plt.subplots(figsize=(10, 4))
                    sns.lineplot(x=serie.index, y=serie.values, marker='o', color='red')
                    plt.title("Evolução de Casos de Hanseníase - TO")
                    st.pyplot(fig)
                else:
                    st.error("Dados insuficientes para criar a série temporal.")
            else:
                st.error(f"Coluna de data não encontrada. Colunas disponíveis: {list(df.columns)}")
        else:
            st.error("Não foi possível obter dados. O servidor do DATASUS (FTP) pode estar instável. Tente novamente em instantes.")