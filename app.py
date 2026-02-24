import streamlit as st
import pandas as pd
import pymannkendall as mk
import matplotlib.pyplot as plt
import seaborn as sns
import re

st.set_page_config(page_title="Analisador Epidemiológico TabNet", layout="wide")

st.title("📊 Analisador de Tendências (Padrão TabNet/DATASUS)")
st.markdown("Esta versão processa tabelas com municípios nas linhas e períodos nas colunas.")

def processar_tabela_tabnet(df):
    # 1. Identifica a coluna de Município (Geralmente a primeira)
    col_geo = df.columns[0]
    
    # 2. Identifica colunas de tempo (ex: 2014/Jan, 2015, etc)
    # Procuramos colunas que começam com 4 dígitos
    cols_tempo = [c for c in df.columns if re.match(r'^\d{4}', str(c))]
    
    if not cols_tempo:
        return None, None, None

    # 3. "Derrete" a tabela (Melt) para ficar vertical
    df_long = df.melt(id_vars=[col_geo], value_vars=cols_tempo, 
                      var_name='Periodo', value_name='Casos')
    
    # 4. Limpeza: Traço '-' vira 0, e remove espaços
    df_long['Casos'] = df_long['Casos'].astype(str).str.replace('-', '0').str.replace('.', '')
    df_long['Casos'] = pd.to_numeric(df_long['Casos'], errors='coerce').fillna(0)
    
    # 5. Extrai o ANO (Pega os primeiros 4 dígitos de '2014/Jan')
    df_long['Ano'] = df_long['Periodo'].astype(str).str[:4].astype(int)
    
    return col_geo, df_long

uploaded_file = st.file_uploader("Upload do arquivo (CSV ou Excel)", type=['csv', 'xlsx'])

if uploaded_file:
    try:
        # Carregamento com separador flexível para o padrão do TabNet (ponto e vírgula)
        try:
            df_raw = pd.read_csv(uploaded_file, sep=None, engine='python', encoding='ISO-8859-1')
        except:
            uploaded_file.seek(0)
            df_raw = pd.read_csv(uploaded_file, sep=None, engine='python', encoding='utf-8')

        # Limpeza de linhas de Total e lixo de rodapé
        df_raw = df_raw[~df_raw.iloc[:, 0].astype(str).str.contains('Total|TOTAL|Incompleto|Fonte|Gerado', na=False)]
        
        col_geo, df_processado = processar_tabela_tabnet(df_raw)
        
        if df_processado is not None:
            # Interface de Filtro
            st.sidebar.header("📍 Localidade")
            lista_locais = sorted(df_processado[col_geo].unique().tolist())
            local = st.sidebar.selectbox("Selecione o Município/Estado:", lista_locais)
            
            # Filtra e agrupa por ano
            df_final = df_processado[df_processado[col_geo] == local]
            serie = df_final.groupby('Ano')['Casos'].sum().sort_index()
            
            # Filtro de período (ex: 2015-2024 conforme seu estudo)
            serie = serie[(serie.index >= 2014) & (serie.index <= 2024)]

            if len(serie) >= 3:
                # ESTATÍSTICA HAMED & RAO
                res = mk.hamed_rao_modification_test(serie)
                
                st.subheader(f"Análise de Tendência: {local}")
                c1, c2, c3 = st.columns(3)
                c1.metric("Tendência", res.trend.upper())
                c2.metric("P-Valor", f"{res.p:.4f}")
                c3.metric("Soma no Período", int(serie.sum()))

                # Gráfico
                fig, ax = plt.subplots(figsize=(10, 4))
                sns.lineplot(x=serie.index, y=serie.values, marker='o', color='#e67e22', linewidth=2.5)
                plt.title(f"Evolução Anual em {local}")
                plt.ylabel("Nº de Notificações")
                plt.xlabel("Ano")
                plt.grid(True, linestyle=':', alpha=0.6)
                st.pyplot(fig)
                
                if res.p < 0.05:
                    st.success(f"Significância Estatística: A tendência de {res.trend} é real.")
                else:
                    st.info("Não há tendência clara (estabilidade estatística).")
            else:
                st.warning("Dados insuficientes para o período selecionado.")
        else:
            st.error("Não identifiquei o padrão de anos nas colunas. Verifique se as colunas têm nomes como '2014/Jan'.")

    except Exception as e:
        st.error(f"Erro no processamento: {e}")