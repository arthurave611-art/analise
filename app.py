import streamlit as st
import pandas as pd
from pysus.online_data import SINAN
import pymannkendall as mk
import matplotlib.pyplot as plt
import seaborn as sns

# Configuração da página
st.set_page_config(page_title="Hanseníase TO - Análise Científica", layout="wide")

st.title("📊 Tendência Temporal de Hanseníase em Tocantins (2015-2024)")
st.markdown("""
Esta aplicação reproduz a metodologia de **Mann-Kendall (Hamed e Rao)** para análise de 
séries temporais de saúde pública, adaptada para os dados de Hanseníase de Tocantins.
""")

@st.cache_data
def carregar_dados_sinan():
    """Função para extrair dados do DATASUS via PySUS com tratamento de erro de parâmetro"""
    try:
        # Tentativa 1: Usando lista no parâmetro states (comum em versões recentes)
        arquivos = SINAN.download('HANS', states=['TO'])
        df = SINAN.to_dataframe(arquivos)
    except TypeError:
        try:
            # Tentativa 2: Usando o parâmetro state no singular
            arquivos = SINAN.download('HANS', state='TO')
            df = SINAN.to_dataframe(arquivos)
        except Exception as e:
            st.error(f"Erro ao tentar baixar dados: {e}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Erro na conexão com o DATASUS: {e}")
        return pd.DataFrame()
    
    # Padroniza os nomes das colunas para maiúsculo
    if not df.empty:
        df.columns = [c.upper() for c in df.columns]
    return df

# Menu lateral
st.sidebar.header("Painel de Controle")

if st.sidebar.button("Executar Análise Completa"):
    with st.spinner("Conectando ao SINAN/DATASUS (pode demorar dependendo do servidor gov)..."):
        df_bruto = carregar_dados_sinan()
        
        if not df_bruto.empty:
            # Tratamento de datas
            df_bruto['DT_NOTIFIC'] = pd.to_datetime(df_bruto['DT_NOTIFIC'], errors='coerce')
            df_bruto['ANO'] = df_bruto['DT_NOTIFIC'].dt.year
            
            # Filtro do recorte temporal (2015 a 2024)
            df_filtrado = df_bruto[(df_bruto['ANO'] >= 2015) & (df_bruto['ANO'] <= 2024)]
            
            # Agrupamento por ano
            serie_temporal = df_filtrado.groupby('ANO').size()
            
            if len(serie_temporal) > 1:
                st.subheader("📈 Resultados da Tendência (Mann-Kendall)")
                
                # Teste de Hamed e Rao
                res = mk.hamed_rao_modification_test(serie_temporal)
                
                c1, c2, c3 = st.columns(3)
                c1.metric("Tendência", res.trend)
                c2.metric("P-Valor", f"{res.p:.4f}")
                c3.metric("Casos Totais", df_filtrado.shape[0])
                
                if res.p < 0.05:
                    st.success("Tendência estatisticamente significativa.")
                else:
                    st.warning("Sem tendência clara detectada (p > 0.05).")

                # Visualização
                st.subheader("🖼️ Gráfico de Evolução Temporal")
                fig, ax = plt.subplots(figsize=(12, 5))
                sns.lineplot(x=serie_temporal.index, y=serie_temporal.values, marker='o', color='darkblue', ax=ax)
                ax.set_title("Casos de Hanseníase Notificados em Tocantins", fontsize=14)
                plt.grid(True, linestyle='--', alpha=0.6)
                st.pyplot(fig)
                
                with st.expander("Ver Tabela de Dados"):
                    st.write(serie_temporal)
            else:
                st.error("Dados insuficientes para calcular a tendência.")
        else:
            st.error("Não foi possível carregar os dados. O servidor do DATASUS pode estar offline ou instável agora.")