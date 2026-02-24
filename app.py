import streamlit as st
import pandas as pd
import pymannkendall as mk
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Analisador Epidemiológico Universal", layout="wide")

st.title("📊 Analisador de Tendências Temporais (Hamed & Rao)")
st.markdown("Faça o upload de qualquer tabela do TabNet/DATASUS para analisar a evolução da doença.")

# --- MOTOR DE IDENTIFICAÇÃO ---
def identificar_estrutura(df):
    col_geo = None
    col_ano = None
    col_casos = None
    
    keywords_geo = ['MUNIC', 'ESTADO', 'UF', 'REGIAO', 'PAIS', 'CAPITAL', 'CIDADE', 'LOCAL']
    keywords_ano = ['ANO', 'NOTIF', 'PERIODO', 'TEMPO']

    for col in df.columns:
        c_up = str(col).upper()
        if any(k in c_up for k in keywords_geo): col_geo = col
        if any(k in c_up for k in keywords_ano): col_ano = col
    
    # A coluna de casos é a primeira numérica que sobra
    for col in df.columns:
        if col not in [col_geo, col_ano]:
            if pd.to_numeric(df[col], errors='coerce').notnull().all():
                col_casos = col
                break
    return col_geo, col_ano, col_casos

# --- INTERFACE ---
uploaded_file = st.file_uploader("Upload do arquivo (CSV ou Excel)", type=['csv', 'xlsx'])

if uploaded_file:
    try:
        # Carregamento com tratamento de erro de encoding
        try:
            df = pd.read_csv(uploaded_file, sep=None, engine='python', encoding='ISO-8859-1')
        except:
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file, sep=None, engine='python', encoding='utf-8')
        
        # Limpeza básica de Totais
        df = df[~df.iloc[:, 0].astype(str).str.contains('Total|TOTAL|Incompleto', na=False)]
        
        c_geo, c_ano, c_casos = identificar_estrutura(df)
        
        if c_ano and c_casos:
            st.sidebar.header("📍 Filtros de Análise")
            
            # Se houver coluna geográfica, permite filtrar
            if c_geo:
                locais = ["Todos"] + sorted(df[c_geo].unique().tolist())
                selecao = st.sidebar.selectbox(f"Selecione o {c_geo}:", locais)
                if selecao != "Todos":
                    df = df[df[c_geo] == selecao]
            
            # Tratamento final dos dados
            df[c_ano] = pd.to_numeric(df[c_ano], errors='coerce')
            df[c_casos] = pd.to_numeric(df[c_casos], errors='coerce')
            df = df.dropna(subset=[c_ano, c_casos])
            
            serie = df.groupby(c_ano)[c_casos].sum().sort_index()

            if len(serie) > 2:
                # ESTATÍSTICA
                res = mk.hamed_rao_modification_test(serie)
                
                # Exibição
                st.subheader(f"Tendência para: {selecao if c_geo and selecao != 'Todos' else 'Área Total'}")
                m1, m2, m3 = st.columns(3)
                m1.metric("Tendência", res.trend.upper())
                m2.metric("P-Valor", f"{res.p:.4f}")
                m3.metric("Total de Casos", int(serie.sum()))

                # GRÁFICO
                fig, ax = plt.subplots(figsize=(10, 4))
                sns.lineplot(x=serie.index, y=serie.values, marker='o', color='#2c3e50', linewidth=2)
                plt.title(f"Série Temporal de {c_casos}", fontsize=12)
                plt.grid(True, linestyle=':', alpha=0.6)
                st.pyplot(fig)
                
                # Interpretação
                if res.p < 0.05:
                    st.success(f"Estatisticamente significativo: Existe uma tendência de {res.trend}.")
                else:
                    st.info("As variações são flutuações sem tendência estatística clara.")
            else:
                st.warning("Poucos pontos de dados para realizar o teste de tendência.")
        else:
            st.error("Não identifiquei as colunas necessárias. Verifique se a tabela tem colunas de tempo e quantidade.")
            
    except Exception as e:
        st.error(f"Erro no processamento: {e}")