import streamlit as st
import pandas as pd
import pymannkendall as mk
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np

st.set_page_config(page_title="CalculaAí - Pesquisa Aí", layout="wide")

# --- LÓGICA GEOGRÁFICA ---
MAPA_ESTADOS = {
    '11': 'RO', '12': 'AC', '13': 'AM', '14': 'RR', '15': 'PA', '16': 'AP', '17': 'TO',
    '21': 'MA', '22': 'PI', '23': 'CE', '24': 'RN', '25': 'PB', '26': 'PE', '27': 'AL', '28': 'SE', '29': 'BA',
    '31': 'MG', '32': 'ES', '33': 'RJ', '35': 'SP',
    '41': 'PR', '42': 'SC', '43': 'RS',
    '50': 'MS', '51': 'MT', '52': 'GO', '53': 'DF'
}

def extrair_geo(linha):
    nome = str(linha).strip()
    codigo = re.search(r'^(\d{2})\d*', nome)
    if codigo:
        cod_uf = codigo.group(1)
        estado = MAPA_ESTADOS.get(cod_uf, 'Outros')
        municipio = re.sub(r'^\d+\s*', '', nome)
        reg_cod = cod_uf[0]
        regioes = {'1':'Norte','2':'Nordeste','3':'Sudeste','4':'Sul','5':'Centro-Oeste'}
        return regioes.get(reg_cod, 'Outros'), estado, municipio
    return 'Brasil', 'Brasil', 'Brasil'

def processar_tabnet(df):
    col_geo = df.columns[0]
    cols_tempo = [c for c in df.columns if re.match(r'^\d{4}', str(c))]
    if not cols_tempo: return None
    df_long = df.melt(id_vars=[col_geo], value_vars=cols_tempo, var_name='Periodo', value_name='Casos')
    df_long['Casos'] = df_long['Casos'].astype(str).str.replace('-', '0').str.replace('.', '').str.replace(',', '.')
    df_long['Casos'] = pd.to_numeric(df_long['Casos'], errors='coerce').fillna(0)
    df_long['Ano'] = df_long['Periodo'].astype(str).str[:4].astype(int)
    geos = df_long[col_geo].apply(extrair_geo)
    df_long[['Regiao', 'Estado', 'Municipio']] = pd.DataFrame(geos.tolist(), index=df_long.index)
    return df_long

st.title("📊 Análise de Tendência de Mann-Kendall")
st.markdown("### Padrão: Hamed & Rao (Correção de Autocorrelação)")

file = st.file_uploader("Upload do CSV (Dengue 2014-2023)", type=['csv'])

if file:
    try:
        df_raw = pd.read_csv(file, sep=';', encoding='ISO-8859-1')
        df_raw = df_raw[~df_raw.iloc[:, 0].astype(str).str.contains('Total|TOTAL|Incompleto|Fonte', na=False)]
        df_final = processar_tabnet(df_raw)
        
        if df_final is not None:
            # --- FILTROS ---
            st.sidebar.header("⚙️ Configurações")
            nivel = st.sidebar.radio("Nível Geográfico:", ("País (Total)", "Região", "Estado", "Município"))
            
            if nivel == "País (Total)":
                df_temp = df_final
                local_label = "Brasil"
            elif nivel == "Região":
                r = st.sidebar.selectbox("Região:", sorted(df_final['Regiao'].unique()))
                df_temp = df_final[df_final['Regiao'] == r]; local_label = r
            elif nivel == "Estado":
                e = st.sidebar.selectbox("Estado:", sorted(df_final['Estado'].unique()))
                df_temp = df_final[df_final['Estado'] == e]; local_label = e
            else:
                uf = st.sidebar.selectbox("UF:", sorted(df_final['Estado'].unique()))
                mun = st.sidebar.selectbox("Município:", sorted(df_final[df_final['Estado'] == uf]['Municipio'].unique()))
                df_temp = df_final[df_final['Municipio'] == mun]; local_label = mun

            # --- CUSTOMIZAÇÃO ---
            st.sidebar.header("🎨 Estilo do Gráfico")
            cor_linha = st.sidebar.color_picker("Cor dos Dados", "#34495e")
            cor_trend = st.sidebar.color_picker("Cor da Tendência", "#e74c3c")
            show_points = st.sidebar.checkbox("Mostrar Pontos", True)

            # Agrupar por ano
            serie = df_temp.groupby('Ano')['Casos'].sum().sort_index()

            if len(serie) >= 4:
                # ESTATÍSTICA (REVERSA)
                res_hr = mk.hamed_rao_modification_test(serie)
                res_orig = mk.original_test(serie)
                
                # Extração segura de atributos (Evita o erro 'object has no attribute')
                trend = getattr(res_hr, 'trend', 'no trend')
                h = getattr(res_hr, 'h', False)
                p = getattr(res_hr, 'p', 0.0)
                z = getattr(res_hr, 'z', 0.0)
                tau = getattr(res_orig, 'tau', 0.0)
                slope = getattr(res_hr, 'slope', 0.0)

                # --- TABELA DE RESULTADOS (PADRÃO PESQUISA AÍ) ---
                st.subheader(f"Métricas do Teste - {local_label}")
                df_res = pd.DataFrame({
                    "Métrica": ["Tendência", "h", "Valor-p", "Estatística Z", "Tau de Kendall", "Inclinação de Sen"],
                    "Resultado": [trend, str(h), f"{p:.8f}", f"{z:.8f}", f"{tau:.8f}", f"{slope:.8f}"]
                })
                st.table(df_res)

                # --- GRÁFICO ---
                
                fig, ax = plt.subplots(figsize=(12, 5))
                ax.plot(serie.index, serie.values, color=cor_linha, label='Dados Reais', linewidth=2, marker='o' if show_points else None)
                
                # Reta de Tendência
                x = np.arange(len(serie))
                intercept = np.median(serie.values) - slope * np.median(x)
                y_trend = slope * x + intercept
                ax.plot(serie.index, y_trend, color=cor_trend, linestyle='--', label=f'Tendência ({trend})', linewidth=2)
                
                plt.title(f"Série Temporal e Tendência: {local_label}", fontsize=14)
                plt.xticks(serie.index)
                plt.grid(True, linestyle=':', alpha=0.6)
                plt.legend()
                st.pyplot(fig)
            else:
                st.warning("Poucos dados para análise (mínimo 4 anos).")
    except Exception as e:
        st.error(f"Erro no processamento: {e}")