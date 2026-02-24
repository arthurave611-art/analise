import streamlit as st
import pandas as pd
import pymannkendall as mk
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np

st.set_page_config(page_title="Analista Epidemiológico Pro", layout="wide")

# Tabelas de Apoio Geográfico
MAPA_ESTADOS = {
    '11': 'RO', '12': 'AC', '13': 'AM', '14': 'RR', '15': 'PA', '16': 'AP', '17': 'TO',
    '21': 'MA', '22': 'PI', '23': 'CE', '24': 'RN', '25': 'PB', '26': 'PE', '27': 'AL', '28': 'SE', '29': 'BA',
    '31': 'MG', '32': 'ES', '33': 'RJ', '35': 'SP',
    '41': 'PR', '42': 'SC', '43': 'RS',
    '50': 'MS', '51': 'MT', '52': 'GO', '53': 'DF'
}

MAPA_REGIOES = {
    '1': 'Norte', '2': 'Nordeste', '3': 'Sudeste', '4': 'Sul', '5': 'Centro-Oeste'
}

def extrair_geografia(linha):
    nome = str(linha).strip()
    codigo = re.search(r'^(\d{2})\d*', nome)
    if codigo:
        cod_uf = codigo.group(1)
        regiao = MAPA_REGIOES.get(cod_uf[0], 'Outros')
        estado = MAPA_ESTADOS.get(cod_uf, 'Outros')
        municipio = re.sub(r'^\d+\s*', '', nome)
        return regiao, estado, municipio
    return 'Brasil', 'Brasil', 'Brasil'

def processar_dados(df):
    col_geo = df.columns[0]
    cols_tempo = [c for c in df.columns if re.match(r'^\d{4}', str(c))]
    if not cols_tempo: return None
    
    df_long = df.melt(id_vars=[col_geo], value_vars=cols_tempo, var_name='Periodo', value_name='Casos')
    df_long['Casos'] = df_long['Casos'].astype(str).str.replace('-', '0').str.replace('.', '')
    df_long['Casos'] = pd.to_numeric(df_long['Casos'], errors='coerce').fillna(0)
    df_long['Ano'] = df_long['Periodo'].astype(str).str[:4].astype(int)
    
    geos = df_long[col_geo].apply(extrair_geografia)
    df_long[['Regiao', 'Estado', 'Municipio']] = pd.DataFrame(geos.tolist(), index=df_long.index)
    return df_long

st.title("📊 Análise de Tendência de Mann-Kendall")
st.markdown("Ferramenta universal para análise de séries temporais epidemiológicas.")

uploaded_file = st.file_uploader("Arraste seu arquivo CSV do TabNet aqui", type=['csv'])

if uploaded_file:
    try:
        df_raw = pd.read_csv(uploaded_file, sep=';', encoding='ISO-8859-1')
        df_raw = df_raw[~df_raw.iloc[:, 0].astype(str).str.contains('Total|TOTAL|Incompleto|Fonte', na=False)]
        df_final = processar_dados(df_raw)
        
        if df_final is not None:
            # --- FILTROS LATERAIS ---
            st.sidebar.header("🗺️ Localidade")
            nivel = st.sidebar.radio("Nível Geográfico:", ("País (Total)", "Região", "Estado", "Município"))
            
            if nivel == "País (Total)":
                df_temp = df_final
                label = "Brasil"
            elif nivel == "Região":
                reg = st.sidebar.selectbox("Região:", sorted(df_final['Regiao'].unique()))
                df_temp = df_final[df_final['Regiao'] == reg]
                label = reg
            elif nivel == "Estado":
                est = st.sidebar.selectbox("Estado:", sorted(df_final['Estado'].unique()))
                df_temp = df_final[df_final['Estado'] == est]
                label = est
            else:
                uf = st.sidebar.selectbox("UF:", sorted(df_final['Estado'].unique()))
                mun = st.sidebar.selectbox("Município:", sorted(df_final[df_final['Estado'] == uf]['Municipio'].unique()))
                df_temp = df_final[df_final['Municipio'] == mun]
                label = mun

            serie = df_temp.groupby('Ano')['Casos'].sum().sort_index()
            # Focar no período 2014-2023 (conforme arquivo enviado)
            serie = serie[(serie.index >= 2014) & (serie.index <= 2024)]

            if len(serie) > 3:
                # CÁLCULO MANN-KENDALL (HAMED & RAO)
                res = mk.hamed_rao_modification_test(serie)
                
                # --- MÉTRICAS ---
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Tendência", res.trend.capitalize())
                c2.metric("P-Valor", f"{res.p:.4f}")
                c3.metric("Z-Score", f"{res.z:.2f}")
                c4.metric("Total de Casos", int(serie.sum()))

                # --- GRÁFICO PERSONALIZADO ---
                [Image of a time series plot showing Mann-Kendall trend line and Sen's slope]
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Dados reais (pontos e linha fina)
                sns.lineplot(x=serie.index, y=serie.values, marker='o', markersize=8, 
                             color='#2c3e50', label='Casos Notificados', ax=ax, linewidth=1, alpha=0.7)
                
                # Linha de Tendência (Sen's Slope)
                # Equação: y = slope * (x - x0) + intercept
                x = np.arange(len(serie))
                y_slope = res.slope * x + serie.values[0] # Estimativa visual da reta
                
                plt.plot(serie.index, y_slope, color='#e74c3c', linestyle='--', 
                         linewidth=2, label=f'Tendência ({res.trend})')

                # Estética do Gráfico
                ax.set_title(f"Evolução Temporal e Tendência em: {label}", fontsize=16, pad=20)
                ax.set_ylabel("Nº de Notificações", fontsize=12)
                ax.set_xlabel("Ano", fontsize=12)
                plt.xticks(serie.index)
                plt.grid(True, linestyle=':', alpha=0.6)
                plt.legend(frameon=True, facecolor='white')
                
                st.pyplot(fig)
                
                # Interpretação
                if res.p < 0.05:
                    st.success(f"📢 Análise Científica: Existe uma tendência de **{res.trend}** estatisticamente significativa para {label}.")
                else:
                    st.info(f"📢 Análise Científica: Não foi detectada tendência significativa (estabilidade) em {label}.")
            else:
                st.warning("Dados insuficientes para rodar o teste estatístico.")
    except Exception as e:
        st.error(f"Ocorreu um erro: {e}")