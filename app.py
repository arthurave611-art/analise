import streamlit as st
import pandas as pd
from pysus.online_data import SINAN
import pymannkendall as mk
import matplotlib.pyplot as plt
import seaborn as sns

# Configuração da página para ocupar a tela inteira
st.set_page_config(page_title="Hanseníase TO - Análise Científica", layout="wide")

st.title("📊 Tendência Temporal de Hanseníase em Tocantins (2015-2024)")
st.markdown("""
Esta aplicação reproduz a metodologia de **Mann-Kendall (Hamed e Rao)** para análise de 
séries temporais de saúde pública, focando nos dados de Hanseníase do estado do Tocantins.
""")

@st.cache_data
def carregar_dados_sinan():
    """Função para extrair dados do DATASUS via PySUS"""
    try:
        # Correção do parâmetro: 'state' no singular é o padrão atual do PySUS para SINAN
        arquivos = SINAN.download('HANS', state='TO')
        df = SINAN.to_dataframe(arquivos)
        
        # Padroniza os nomes das colunas para maiúsculo para evitar erros de busca
        df.columns = [c.upper() for c in df.columns]
        return df
    except Exception as e:
        st.error(f"Erro na conexão com o DATASUS: {e}")
        return pd.DataFrame()

# Menu lateral
st.sidebar.header("Painel de Controle")
st.sidebar.info("Clique no botão abaixo para iniciar a coleta de dados em tempo real.")

if st.sidebar.button("Executar Análise Completa"):
    with st.spinner("Conectando ao SINAN/DATASUS..."):
        df_bruto = carregar_dados_sinan()
        
        if not df_bruto.empty:
            # Tratamento de datas
            # A coluna DT_NOTIFIC é a data da notificação do caso
            df_bruto['DT_NOTIFIC'] = pd.to_datetime(df_bruto['DT_NOTIFIC'], errors='coerce')
            df_bruto['ANO'] = df_bruto['DT_NOTIFIC'].dt.year
            
            # Filtro do recorte temporal (2015 a 2024)
            df_filtrado = df_bruto[(df_bruto['ANO'] >= 2015) & (df_bruto['ANO'] <= 2024)]
            
            # Agrupamento por ano para criar a série temporal
            serie_temporal = df_filtrado.groupby('ANO').size()
            
            if len(serie_temporal) > 1:
                # --- RESULTADOS ESTATÍSTICOS ---
                st.subheader("📈 Resultados da Tendência (Mann-Kendall)")
                
                # Teste de Hamed e Rao (específico para dados com autocorrelação, como no artigo)
                res = mk.hamed_rao_modification_test(serie_temporal)
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Tendência Detectada", res.trend)
                c2.metric("P-Valor (Significância)", f"{res.p:.4f}")
                c3.metric("Total de Casos Analisados", df_filtrado.shape[0])
                
                # Interpretação científica
                if res.p < 0.05:
                    st.success("A tendência é estatisticamente significativa.")
                else:
                    st.warning("Não há evidência estatística de tendência clara (p > 0.05).")

                # --- VISUALIZAÇÃO GRÁFICA ---
                st.subheader("🖼️ Gráfico de Evolução Temporal")
                fig, ax = plt.subplots(figsize=(12, 5))
                sns.lineplot(x=serie_temporal.index, y=serie_temporal.values, marker='o', color='teal', ax=ax)
                
                # Estilização do gráfico
                ax.set_title("Número de Casos de Hanseníase Notificados em Tocantins", fontsize=14)
                ax.set_xlabel("Ano de Notificação")
                ax.set_ylabel("Quantidade de Casos")
                plt.grid(True, linestyle='--', alpha=0.6)
                
                st.pyplot(fig)
                
                # --- DADOS CLÍNICOS (Diferencial para Medicina/Semiologia) ---
                with st.expander("Ver Detalhes dos Dados por Ano"):
                    st.write(serie_temporal)
            else:
                st.error("Dados insuficientes para calcular a tendência.")
        else:
            st.error("A base de dados retornou vazia. Tente novamente em instantes.")