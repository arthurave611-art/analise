import streamlit as st
import pandas as pd
import pymannkendall as mk
import matplotlib.pyplot as plt
import seaborn as sns
import re
import numpy as np

st.set_page_config(page_title="Bioestatística Avançada - Pesquisa Aí", layout="wide")

# --- DICIONÁRIOS DE APOIO IBGE ---
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

# --- FUNÇÕES DE PROCESSAMENTO ---
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
    df_long['Casos'] = df_long['Casos'].astype(str).str.replace('-', '0').str.replace('.', '').str.replace(',', '.')
    df_long['Casos'] = pd.to_numeric(df_long['Casos'], errors='coerce').fillna(0)
    df_long['Ano'] = df_long['Periodo'].astype(str).str[:4].astype(int)
    
    geos = df_long[col_geo].apply(extrair_geografia)
    df_long[['Regiao', 'Estado', 'Municipio']] = pd.DataFrame(geos.tolist(), index=df_long.index)
    return df_long

# --- INTERFACE PRINCIPAL ---
st.title("📊 Análise de Tendência de Mann-Kendall")
st.markdown("### Método: Hamed & Rao (Correção de Autocorrelação)")

uploaded_file = st.file_uploader("Upload do arquivo CSV (Padrão DataSUS/TabNet)", type=['csv'])

if uploaded_file:
    try:
        df_raw = pd.read_csv(uploaded_file, sep=';', encoding='ISO-8859-1')
        df_raw = df_raw[~df_raw.iloc[:, 0].astype(str).str.contains('Total|TOTAL|Incompleto|Fonte', na=False)]
        df_final = processar_dados(df_raw)
        
        if df_final is not None:
            # --- SIDEBAR: FILTROS E CUSTOMIZAÇÃO ---
            st.sidebar.header("🗺️ Configurações de Dados")
            nivel = st.sidebar.radio("Nível Observacional:", ("País (Total)", "Região", "Estado", "Município"))
            
            if nivel == "País (Total)":
                df_temp = df_final
                label_local = "Brasil"
            elif nivel == "Região":
                reg = st.sidebar.selectbox("Selecione a Região:", sorted(df_final['Regiao'].unique()))
                df_temp = df_final[df_final['Regiao'] == reg]
                label_local = reg
            elif nivel == "Estado":
                est = st.sidebar.selectbox("Selecione o Estado:", sorted(df_final['Estado'].unique()))
                df_temp = df_final[df_final['Estado'] == est]
                label_local = est
            else:
                uf = st.sidebar.selectbox("Selecione a UF:", sorted(df_final['Estado'].unique()))
                mun = st.sidebar.selectbox("Selecione o Município:", sorted(df_final[df_final['Estado'] == uf]['Municipio'].unique()))
                df_temp = df_final[df_final['Municipio'] == mun]
                label_local = mun

            # Customização do Gráfico
            st.sidebar.header("🎨 Customização do Gráfico")
            cor_linha = st.sidebar.color_picker("Cor dos Dados", "#2c3e50")
            cor_tendencia = st.sidebar.color_picker("Cor da Tendência", "#e74c3c")
            marker_style = st.sidebar.selectbox("Estilo do Marcador", ["o", "s", "D", "^", "v", "x"])
            line_width = st.sidebar.slider("Espessura da Linha", 1.0, 5.0, 2.0)
            show_grid = st.sidebar.checkbox("Mostrar Grade", True)

            # Agrupamento e Análise
            serie = df_temp.groupby('Ano')['Casos'].sum().sort_index()

            if len(serie) > 3:
                # CÁLCULOS ESTATÍSTICOS (DUPLO CHECK PARA EVITAR ERRO DE ATRIBUTO)
                res_hr = mk.hamed_rao_modification_test(serie)
                res_orig = mk.original_test(serie)
                
                # --- TABELA DE RESULTADOS (PADRÃO SOLICITADO) ---
                st.subheader(f"Métricas do Teste - {label_local}")
                
                # Captura segura do Tau e Slope
                tau_val = getattr(res_hr, 'tau', getattr(res_orig, 'tau', 0))
                slope_val = getattr(res_hr, 'slope', 0)
                
                metrics_data = {
                    "Métrica": ["Tendência", "h", "Valor-p", "Estatística Z", "Tau de Kendall", "Inclinação de Sen"],
                    "Resultado": [
                        res_hr.trend, 
                        str(res_hr.h), 
                        f"{res_hr.p:.8f}", 
                        f"{res_hr.z:.8f}", 
                        f"{tau_val:.8f}", 
                        f"{slope_val:.8f}"
                    ]
                }
                st.table(pd.DataFrame(metrics_data))

                # --- GRÁFICO CUSTOMIZÁVEL ---
                fig, ax = plt.subplots(figsize=(12, 6))
                
                # Plot dos dados reais
                ax.plot(serie.index, serie.values, marker=marker_style, markersize=8, 
                        color=cor_linha, label='Dados Observados', linewidth=line_width)
                
                # Cálculo da Reta de Tendência (Sen's Slope)
                x_idx = np.arange(len(serie))
                # Intercepto baseado na mediana para cruzar os dados corretamente
                intercept = np.median(serie.values) - slope_val * np.median(x_idx)
                y_trend = slope_val * x_idx + intercept
                
                ax.plot(serie.index, y_trend, color=cor_tendencia, linestyle='--', 
                        linewidth=line_width + 1, label=f'Reta de Tendência (Slope: {slope_val:.2f})')

                ax.set_title(f"Série Temporal e Tendência: {label_local}", fontsize=16)
                ax.set_ylabel("Quantidade de Casos")
                ax.set_xlabel("Ano")
                plt.xticks(serie.index)
                if show_grid: ax.grid(True, linestyle=':', alpha=0.6)
                plt.legend()
                
                st.pyplot(fig)
                
                # Botão para Download dos Dados
                csv_out = serie.reset_index().to_csv(index=False).encode('utf-8')
                st.download_button("📥 Baixar Dados da Série (CSV)", csv_out, "serie_temporal.csv", "text/csv")
                
            else:
                st.warning("⚠️ Série temporal muito curta para o teste estatístico (mínimo 4 pontos).")
                
    except Exception as e:
        st.error(f"❌ Erro Crítico: {e}")
        st.info("Dica: Verifique se o separador do seu CSV é ponto e vírgula (;).")