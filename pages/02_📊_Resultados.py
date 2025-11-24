import streamlit as st
import pandas as pd
from joblib import load
import plotly.express as px
import plotly.graph_objects as go
from plots import (
    calcular_metricas_fraude,
    calcular_e_plotar_impacto,
    plot_taxa_fraude,
    plot_proporcao_fraudes,
    plot_radar_metricas,
    markdown
)

# ---------------- Configuração da Página ----------------
st.set_page_config(page_title="Relatório de Detecção de Fraudes", layout="wide")

# ---------------- Carregamento da Matriz ----------------
_, matriz = load("objects/metricas.pkl")

# ---------------- Barra Lateral ----------------
with st.sidebar:
    st.title("📈 Navegação")
    visualizacao = st.radio(
        ":green[Selecione a visualização]", ("Métricas de Desempenho", "Impacto Financeiro", "ROI", "Proporção de Fraudes"), 
        label_visibility="visible"
         )
    if visualizacao == "ROI":
        custo_projeto = st.number_input("Custo do Projeto (R$):", min_value=1000, value=80000, step=1000)
    visualizar = st.button("Visualizar", use_container_width=True,
                            help="Clique para gerar a visualização selecionada.", type='primary')

    st.markdown(markdown, unsafe_allow_html=True)

# ---------------- Conteúdo Principal ----------------
if visualizar:

    # ----------- MÉTRICAS DE DESEMPENHO -----------
    if visualizacao == "Métricas de Desempenho":        
        st.header("📊 Métricas de Desempenho do Modelo")
        progress = st.progress(50, text="Calculando Métricas de Desempenho...")

        metricas = calcular_metricas_fraude(matriz)

        col1, col2, col3 = st.columns([0.55, 0.15, 0.3], border=True)
        with col1:

            st.plotly_chart(plot_radar_metricas(metricas), use_container_width=True,
                             config={"displayModeBar": False, 'height': 700})        
        
            
        with col2:
            st.markdown("### KPIs Principais")
            st.metric("1 - Precisão dos Alertas (%) ", metricas["precisao_alerta"])
            st.metric("2 - Recall de Fraudes (%) "  , metricas["recall_fraude"])
            st.metric("3 - Falsos Positivos (%) ", metricas["taxa_falsos_positivos"])            
            
            st.metric("4 - Falsos Negativos (%) ", metricas["taxa_falsos_negativos"])        
            st.metric("5 - Taxa de Alertas (%) ", metricas["taxa_alerta"])
            st.metric("6 - Taxa Não Alertada (%) ", metricas["taxa_nao_alerta"])
            st.metric("7 - F1-Score (%) ", metricas["f1_score"])            
                 
        with col3:
            st.markdown("""
        ## ❓ O que significa?
        ##### 1) **Precisão dos Alertas:**
          - *entre os alertas gerados, quantos realmente eram fraudes.*
        ##### 2) **Recall de Fraudes:**
          - *entre todas as fraudes reais, quantas foram capturadas pelo modelo.*
        ##### 3) **Taxa de Falsos Positivos:**
         - *clientes legítimos sinalizados incorretamente.*
        ##### 4) **Taxa de Falsos Negativos:**
        -  *fraudes que passaram despercebidas.*
        ##### 5) **Taxa de Alertas:**
        -  *proporção de transações sinalizadas como suspeitas.*
        ##### 6) **Taxa Não Alertada:**
        - *proporção de transações consideradas seguras.*
        ##### 7) - **F1-Score:**
         - *equilíbrio entre precisão e recall.*
        """)
        progress.progress(100, text="Cálculo Concluído!")
                
            

    # ----------- IMPACTO FINANCEIRO -----------
    if visualizacao == "Impacto Financeiro":        
        st.header("💰 Impacto Financeiro da Detecção de Fraudes")
        progress = st.progress(50, text="Calculando Impacto Financeiro...")        

        df_impacto, fig_impacto = calcular_e_plotar_impacto(matriz, valor_medio_emprestimo=1200, taxa_juros=0.29)

        col1, col2 = st.columns([0.65, 0.35], border=True)
        with col1:

            st.plotly_chart(fig_impacto, use_container_width=True)
        with col2:
            st.markdown("""
        ### ❓ O que significa?
        Este gráfico mostra como o modelo afeta diretamente os resultados financeiros:
        - **Ganho com bons clientes**: receita gerada por clientes legítimos aprovados.  
        - **Perda por fraudes aprovadas**: prejuízo causado por fraudes que passaram.  
        - **Perda por bons reprovados**: receita perdida por clientes legítimos rejeitados.  
        - **Economia por fraudes reprovadas**: valor economizado ao bloquear fraudes corretamente.  
        """)
        progress.progress(100, text="Cálculo Concluído!")
            

    # ----------- ROI -----------
    if visualizacao == "ROI":        
        st.header("📈 ROI da Detecção de Fraudes", divider="green")        
        progress = st.progress(50, text="Calculando ROI...")

        df_impacto, _ = calcular_e_plotar_impacto(matriz, valor_medio_emprestimo=1200, taxa_juros=0.29)
        economia = df_impacto.loc[df_impacto["Cenário"] == "Economia por fraudes reprovadas", "Valor (R$)"].values[0]                
        retorno_liquido = economia - custo_projeto
        roi_percentual = retorno_liquido / custo_projeto * 100

        fig_waterfall = go.Figure(go.Waterfall(
            name="ROI",
            orientation="v",
            measure=["relative", "relative", "total"],
            x=["Economia com Fraudes Detectadas", "Custo do Projeto", "ROI Líquido"],
            y=[economia, -custo_projeto, retorno_liquido],
            connector={"line": {"color": "gray"}},
            increasing={"marker": {"color": "#2ecc71"}},  # verde para economia
            decreasing={"marker": {"color": "#e74c3c"}},  # vermelho para custo
            totals={"marker": {"color": "#2ecc71"}}       # azul para ROI líquido
        ))
        fig_waterfall.update_layout(
            title="💰 Retorno sobre Investimento (Waterfall)",
            yaxis_title="Valor (R$)",
            xaxis_title="Componentes",
            height=400
        )
                    

        col1, col2 = st.columns([0.65, 0.35], border=True)
    
        with col1:
            st.plotly_chart(fig_waterfall, use_container_width=True)        

            st.markdown("<hr style='border: 1px solid #2ecc71'>", unsafe_allow_html=True)
            st.markdown("### 💰 Interpretação Financeira")
            st.markdown("""
            Este gráfico demonstra o **retorno financeiro obtido com a detecção de fraudes**, comparando os ganhos com os custos do projeto.  
            Ele refere-se ao valor que foi **recuperado ou evitado** graças à atuação do modelo:
            - **Economia gerada** pela detecção de fraudes.
            - **Investimento realizado** no projeto.
            - **ROI líquido**, que representa o saldo positivo da iniciativa.
            """)    
            
        with col2:
                        
            st.markdown("""
            ### ❓ O que significa?
            O ROI (Retorno sobre Investimento) mostra se o projeto compensa financeiramente:
            - **Waterfall**: O gráfico ao lado mostra o fluxo de valores até o ROI líquido.
            - **Economia com fraudes detectadas**: valor recuperado.  
            - **Custo do projeto**: investimento necessário.  
            - **ROI líquido**: diferença entre economia e custo.  
            """)
            
        
            st.markdown("<hr style='border: 1px solid #2ecc71'>", unsafe_allow_html=True)            
            st.markdown("## Resumo Financeiro:")
            st.metric("Retorno Líquido (R$)", f"{(retorno_liquido ):,.2f}")
            st.metric("Economia Total (R$) -> Excluindo-se os custos do projeto ", f"{(economia ):,.2f}")
            st.metric("ROI (%)", f"{roi_percentual:.2f}%")

        progress.progress(100, text="Cálculo Concluído!")
            

    # ----------- PROPORÇÃO DE FRAUDES -----------
    if visualizacao == "Proporção de Fraudes":        
        st.header("📉 Proporção de Fraudes Detectadas vs Não Detectadas", divider="green")
        progress = st.progress(50, text="Calculando Proporção de Fraudes...")
        col1, col2 = st.columns([0.35, 0.65], border=True)
        with col1:
            st.markdown("""
            ### ❓ O que significa?
            Este gráfico mostra a proporção de fraudes que o modelo conseguiu capturar em relação às que passaram despercebidas.
            - **Fraudes Detectadas**: sucesso do modelo.  
            - **Fraudes Não Detectadas**: risco residual que ainda precisa ser mitigado.  
            """)
                                    
            st.markdown("### 📌 Por que essa métrica importa?")
            st.markdown("""                        
            #### Com ela é possível entender:
            - **Quão bem o modelo está performando** na identificação de fraudes.
            - **O nível de risco residual**, ou seja, fraudes que ainda escapam à detecção.               
            - **Áreas para melhoria do modelo** e estratégias de mitigação de risco.
            """)

            
        with col2:
            st.plotly_chart(plot_proporcao_fraudes(matriz), use_container_width=True)

        progress.progress(100, text="Cálculo Concluído!")