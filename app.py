import streamlit as st
import pandas as pd
from pysus.online_data import SINAN
import pymannkendall as mk
import matplotlib.pyplot as plt
import seaborn as sns

# Configuração da página
st.set_page_config(page_title="Hanseníase TO - Análise Estatística", layout="wide")

st.title("📊 Análise de Tendência: Hanseníase em Tocantins (2015-2024)")
st.markdown("""
Esta aplicação utiliza a metodologia de **Mann-Kendall com modificação de Hamed e Rao**, 
conforme aplicada no estudo de tendências temporais do Censo Escolar.
""")

@st.cache_data
def carregar_dados_sinan():
    # Extração de dados de Hanseníase (HANS) para o estado de Tocantins (TO)
    try:
        arquivos = SINAN.download('HANS', states='TO')
        df = SINAN.to_dataframe(arquivos)
        # Padroniza colunas para maiúsculo para evitar erros de referência
        df.columns = [c.upper() for c in df.columns]
        return df
    except Exception as e:
        st.error(f"Erro ao conectar ao DATASUS: {e}")
        return pd.DataFrame()

# Sidebar para controlo
st.sidebar.header("Configurações da Análise")
if st.sidebar.button("Extrair e Processar Dados"):
    with st.spinner("Descarregando dados do SINAN (isto pode demorar alguns minutos)..."):
        df = carregar_dados_sinan()
        
        if not df.empty:
            # Tratamento de Datas
            df['DT_NOTIFIC'] = pd.to_datetime(df['DT_NOTIFIC'], errors='coerce')
            df['ANO'] = df['DT_NOTIFIC'].dt.year
            
            # Filtro do período (2015 a 2024)
            df_filtrado = df[(df['ANO'] >= 2015) & (df['ANO'] <= 2024)]
            
            # Agrupamento anual (Série Temporal)
            serie_temporal = df_filtrado.groupby('ANO').size()
            
            if len(serie_temporal) > 1:
                # --- Bloco de Estatística (Mann-Kendall) ---
                st.subheader("📈 Resultados da Análise de Tendência")
                
                # Teste de Hamed e Rao (indicado para séries com autocorrelação)
                res = mk.hamed_rao_modification_test(serie_temporal)
                
                col1, col2, col3 = st.columns(3)
                col1.metric("Tendência", res.trend)
                col2.metric("P-Valor", f"{res.p:.4f}")
                col3.metric("Total de Casos (Período)", df_filtrado.shape[0])
                
                if res.p < 0.05:
                    st.success("A tendência é estatisticamente significativa (p < 0.05).")
                else:
                    st.info("Não foi detetada tendência com significância estatística.")

                # --- Visualização ---
                st.subheader("🗺️ Evolução dos Casos por Ano")
                fig, ax = plt.subplots(figsize=(12, 5))
                sns.lineplot(x=serie_temporal.index, y=serie_temporal.values, marker='o', color='darkred', ax=ax)
                ax.set_xlabel("Ano de Notificação")
                ax.set_ylabel("Nº de Casos")
                plt.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                # Mostrar tabela de dados brutos
                with st.expander("Ver dados tabulares"):
                    st.write(serie_temporal)
            else:
                st.warning("Dados insuficientes para realizar o teste de tendência.")
        else:
            st.error("Não foi possível carregar os dados. Verifique a conexão com o DATASUS.")