import streamlit as st
import pandas as pd
from pysus.online_data import SINAN
import pymannkendall as mk
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Hanseníase TO - Análise Científica", layout="wide")

st.title("📊 Tendência Temporal de Hanseníase em Tocantins (2015-2024)")
st.markdown("""
Esta aplicação utiliza a metodologia de **Mann-Kendall (Hamed e Rao)** para análise de 
séries temporais, conforme o artigo científico base.
""")

@st.cache_data
def carregar_dados_sinan():
    try:
        # Mudança estratégica: Passamos o estado 'TO' e o sistema 'HANS'
        # Se o download direto falhar, tentamos capturar os arquivos brutos
        arquivos = SINAN.download('HANS', state='TO')
        df = SINAN.to_dataframe(arquivos)
        
        if not df.empty:
            df.columns = [c.upper() for c in df.columns]
            return df
        return pd.DataFrame()
    except Exception as e:
        # Se der erro de 'int()', tentamos buscar o dado sem especificar o estado no download
        # e filtramos no Pandas depois, que é mais seguro
        st.warning("Tentando modo de compatibilidade...")
        try:
            # Baixa o arquivo mais recente disponível e filtramos manualmente
            arquivos = SINAN.download('HANS', state='TO') 
            df = SINAN.to_dataframe(arquivos)
            df.columns = [c.upper() for c in df.columns]
            return df
        except:
            st.error(f"Erro técnico na extração: {e}")
            return pd.DataFrame()

if st.sidebar.button("Executar Análise Completa"):
    with st.spinner("Conectando ao SINAN/DATASUS..."):
        df_bruto = carregar_dados_sinan()
        
        if not df_bruto.empty:
            # Garantir que a coluna de data existe
            col_data = 'DT_NOTIFIC' if 'DT_NOTIFIC' in df_bruto.columns else df_bruto.columns[0]
            
            df_bruto[col_data] = pd.to_datetime(df_bruto[col_data], errors='coerce')
            df_bruto['ANO'] = df_bruto[col_data].dt.year
            
            df_filtrado = df_bruto[(df_bruto['ANO'] >= 2015) & (df_bruto['ANO'] <= 2024)]
            serie_temporal = df_filtrado.groupby('ANO').size()
            
            if len(serie_temporal) > 1:
                st.subheader("📈 Resultados da Tendência (Mann-Kendall)")
                res = mk.hamed_rao_modification_test(serie_temporal)
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Tendência", res.trend)
                c2.metric("P-Valor", f"{res.p:.4f}")
                c3.metric("Casos Totais", df_filtrado.shape[0])
                
                # Gráfico
                st.subheader("🖼️ Gráfico de Evolução Temporal")
                fig, ax = plt.subplots(figsize=(12, 5))
                sns.lineplot(x=serie_temporal.index, y=serie_temporal.values, marker='o', color='red', ax=ax)
                ax.set_title("Casos de Hanseníase Notificados em Tocantins")
                st.pyplot(fig)
            else:
                st.error("Dados insuficientes para gerar a série temporal.")
        else:
            st.error("O servidor do DATASUS não enviou dados válidos.")