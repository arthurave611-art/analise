import streamlit as st
import pandas as pd
import pymannkendall as mk
import matplotlib.pyplot as plt
import seaborn as sns
import re

st.set_page_config(page_title="Analisador Epidemiológico Expert", layout="wide")

st.title("📊 Análise de Tendência Epidemiológica")
st.markdown("Suporte para tabelas pivoteadas do TabNet/DATASUS com níveis de agregação.")

def limpar_nome_municipio(nome):
    # Remove códigos IBGE (números no início do nome)
    return re.sub(r'^\d+\s*', '', str(nome))

def processar_dados_tabnet(df):
    col_geo = df.columns[0]
    # Identifica colunas de tempo (ex: 2014/Jan ou 2015)
    cols_tempo = [c for c in df.columns if re.match(r'^\d{4}', str(c))]
    
    if not cols_tempo:
        return None, None, None

    # Transforma a tabela de formato largo para longo (vertical)
    df_long = df.melt(id_vars=[col_geo], value_vars=cols_tempo, 
                      var_name='Periodo', value_name='Casos')
    
    # Limpeza de valores (traços e pontos de milhar)
    df_long['Casos'] = df_long['Casos'].astype(str).str.replace('-', '0').str.replace('.', '')
    df_long['Casos'] = pd.to_numeric(df_long['Casos'], errors='coerce').fillna(0)
    
    # Extração do Ano
    df_long['Ano'] = df_long['Periodo'].astype(str).str[:4].astype(int)
    
    # Limpeza do nome da localidade
    df_long['Localidade'] = df_long[col_geo].apply(limpar_nome_municipio)
    
    return 'Localidade', df_long

uploaded_file = st.file_uploader("Upload da tabela (CSV ou Excel)", type=['csv', 'xlsx'])

if uploaded_file:
    try:
        # Carregamento com encoding ISO-8859-1 (padrão TabNet)
        try:
            df_raw = pd.read_csv(uploaded_file, sep=None, engine='python', encoding='ISO-8859-1')
        except:
            uploaded_file.seek(0)
            df_raw = pd.read_csv(uploaded_file, sep=None, engine='python', encoding='utf-8')

        # Remove linhas de Total nativas da tabela para não duplicar na nossa soma
        df_raw = df_raw[~df_raw.iloc[:, 0].astype(str).str.contains('Total|TOTAL|Incompleto', na=False)]
        
        col_geo, df_proc = processar_dados_tabnet(df_raw)
        
        if df_proc is not None:
            st.sidebar.header("🔍 Nível de Análise")
            
            # Opções de filtro: Estado (soma de todos) ou Município individual
            opcoes_locais = ["ESTADO (SOMA TOTAL)"] + sorted(df_proc[col_geo].unique().tolist())
            selecao = st.sidebar.selectbox("Selecione a abrangência:", opcoes_locais)
            
            if selecao == "ESTADO (SOMA TOTAL)":
                serie = df_proc.groupby('Ano')['Casos'].sum().sort_index()
                titulo_grafico = "Tendência Geral do Estado"
            else:
                serie = df_proc[df_proc[col_geo] == selecao].groupby('Ano')['Casos'].sum().sort_index()
                titulo_grafico = f"Tendência em {selecao}"

            # Filtro de período (ajustável conforme necessidade)
            serie = serie[(serie.index >= 2014) & (serie.index <= 2024)]

            if len(serie) >= 3:
                # ESTATÍSTICA: Hamed & Rao (conforme artigo base)
                res = mk.hamed_rao_modification_test(serie)
                
                # Exibição de Métricas
                c1, c2, c3 = st.columns(3)
                c1.metric("Tendência Detectada", res.trend.upper())
                c2.metric("P-Valor (Significância)", f"{res.p:.4f}")
                c3.metric("Total de Notificações", int(serie.sum()))

                # Gráfico Epidemiológico
                
                fig, ax = plt.subplots(figsize=(10, 4))
                sns.lineplot(x=serie.index, y=serie.values, marker='o', color='#2c3e50', linewidth=2.5)
                plt.title(titulo_grafico, fontsize=14)
                plt.ylabel("Nº de Casos")
                plt.xlabel("Ano")
                plt.grid(True, linestyle='--', alpha=0.6)
                st.pyplot(fig)
                
                # Interpretação Clínica/Estatística
                if res.p < 0.05:
                    st.success(f"A análise confirma uma tendência de **{res.trend}** com validade estatística.")
                else:
                    st.info("Não há tendência estatística clara (estabilidade ou variação aleatória).")
            else:
                st.warning("Dados insuficientes para este local no período selecionado.")
        else:
            st.error("Formato de data não reconhecido nas colunas.")

    except Exception as e:
        st.error(f"Erro no processamento: {e}")