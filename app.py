import streamlit as st
import pandas as pd
import pymannkendall as mk
import matplotlib.pyplot as plt
import seaborn as sns
import re

st.set_page_config(page_title="Analista Epidemiológico Pro", layout="wide")

# Mapeamento Geográfico Brasileiro (Baseado no IBGE)
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
    # Tenta pegar o código IBGE (6 dígitos no início)
    codigo = re.search(r'^(\d{2})\d*', nome)
    if codigo:
        cod_uf = codigo.group(1)
        regiao = MAPA_REGIOES.get(cod_uf[0], 'Ignorado')
        estado = MAPA_ESTADOS.get(cod_uf, 'Ignorado')
        municipio = re.sub(r'^\d+\s*', '', nome)
        return regiao, estado, municipio
    
    # Caso especial para "Ignorado - UF"
    if '-' in nome:
        uf_final = nome.split('-')[-1].strip()
        for reg_cod, reg_nome in MAPA_REGIOES.items():
            # Simplificação: se for RO, AC, AM, RR, PA, AP, TO -> Norte
            norte = ['RO', 'AC', 'AM', 'RR', 'PA', 'AP', 'TO']
            if uf_final in norte: return 'Norte', uf_final, nome
        return 'Outros', uf_final, nome
        
    return 'Brasil', 'Brasil', 'Brasil'

def processar_tabnet_completo(df):
    col_geo_original = df.columns[0]
    cols_tempo = [c for c in df.columns if re.match(r'^\d{4}', str(c))]
    
    if not cols_tempo: return None
    
    # Derrete a tabela (Melt)
    df_long = df.melt(id_vars=[col_geo_original], value_vars=cols_tempo, 
                      var_name='Periodo', value_name='Casos')
    
    # Limpeza de números
    df_long['Casos'] = df_long['Casos'].astype(str).str.replace('-', '0').str.replace('.', '')
    df_long['Casos'] = pd.to_numeric(df_long['Casos'], errors='coerce').fillna(0)
    df_long['Ano'] = df_long['Periodo'].astype(str).str[:4].astype(int)
    
    # Aplica a inteligência geográfica
    geos = df_long[col_geo_original].apply(extrair_geografia)
    df_long[['Regiao', 'Estado', 'Municipio']] = pd.DataFrame(geos.tolist(), index=df_long.index)
    
    return df_long

st.title("📊 Analisador de Tendências Epidemiológicas")
st.markdown("Análise de **Hamed & Rao** com filtros por País, Região, Estado ou Município.")

uploaded_file = st.file_uploader("Upload do arquivo do TabNet (Dengue, Hanseníase, etc)", type=['csv'])

if uploaded_file:
    try:
        df_raw = pd.read_csv(uploaded_file, sep=';', encoding='ISO-8859-1')
        # Limpa rodapé
        df_raw = df_raw[~df_raw.iloc[:, 0].astype(str).str.contains('Total|TOTAL|Incompleto|Fonte', na=False)]
        
        df_final = processar_tabnet_completo(df_raw)
        
        if df_final is not None:
            st.sidebar.header("🗺️ Escopo da Análise")
            nivel = st.sidebar.radio("Selecione o Nível Geográfico:", 
                                    ("País (Total)", "Região", "Estado", "Município"))
            
            # Lógica de Filtros Dinâmicos
            if nivel == "País (Total)":
                serie = df_final.groupby('Ano')['Casos'].sum()
                titulo = "Tendência Nacional"
            
            elif nivel == "Região":
                regiao_sel = st.sidebar.selectbox("Escolha a Região:", sorted(df_final['Regiao'].unique()))
                serie = df_final[df_final['Regiao'] == regiao_sel].groupby('Ano')['Casos'].sum()
                titulo = f"Tendência na Região {regiao_sel}"
            
            elif nivel == "Estado":
                estado_sel = st.sidebar.selectbox("Escolha o Estado (UF):", sorted(df_final['Estado'].unique()))
                serie = df_final[df_final['Estado'] == estado_sel].groupby('Ano')['Casos'].sum()
                titulo = f"Tendência no Estado: {estado_sel}"
                
            else: # Município
                # Filtra estado primeiro para facilitar a busca do município
                uf_filtro = st.sidebar.selectbox("Filtrar por UF primeiro:", sorted(df_final['Estado'].unique()))
                mun_lista = sorted(df_final[df_final['Estado'] == uf_filtro]['Municipio'].unique())
                mun_sel = st.sidebar.selectbox("Escolha o Município:", mun_lista)
                serie = df_final[df_final['Municipio'] == mun_sel].groupby('Ano')['Casos'].sum()
                titulo = f"Tendência em {mun_sel} - {uf_filtro}"

            # Execução da Estatística
            serie = serie.sort_index()
            if len(serie) >= 3:
                res = mk.hamed_rao_modification_test(serie)
                
                st.subheader(titulo)
                c1, c2, c3 = st.columns(3)
                c1.metric("Tendência", res.trend.upper())
                c2.metric("P-Valor", f"{res.p:.4f}")
                c3.metric("Total de Casos", int(serie.sum()))

                fig, ax = plt.subplots(figsize=(10, 4))
                sns.lineplot(x=serie.index, y=serie.values, marker='o', color='#c0392b', linewidth=2)
                plt.title(f"Série Temporal: {titulo}")
                plt.grid(True, alpha=0.3)
                st.pyplot(fig)
                
                if res.p < 0.05:
                    st.success(f"Significância estatística confirmada para {res.trend}.")
                else:
                    st.info("As flutuações são estáveis estatisticamente.")
            else:
                st.warning("Dados insuficientes para este nível de filtro.")
    except Exception as e:
        st.error(f"Erro: {e}")