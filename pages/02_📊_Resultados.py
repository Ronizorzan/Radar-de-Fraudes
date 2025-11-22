import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import numpy as np
from plots import *
from joblib import load


#Configuração da página
st.set_page_config(page_title="Relatório Financeiro", layout="wide", initial_sidebar_state="expanded")


# Carrega matriz de confusão
_, matriz = load("objects/metricas.pkl")


# Calcula métricas necessárias para plotagens apartir da matriz de confusão
resultado_xgb = calcular_metricas(matriz)


# Controles da Barra Lateral
with st.sidebar:    
    valor_medio_emprest = np.float32(1000)
    taxa_juros = np.int32(29) / 100  # Taxa de juros padrão de 29%

    st.title("Visualização dos Resultados")
    with st.expander("Configurações das visualizações", expanded=True):
        df_final = pd.read_csv('Fraud_transactions.csv')  # Carrega o DataFrame final do arquivo CSV
        visualizacao = st.radio("Selecione o tipo de visualização", ("Análise Descritiva", "Impacto Financeiro", "Redução da Inadimplência",
                                                                      "Taxa de Aprovação"))
    valor_medio_emprest = st.number_input("Valor médio dos empréstimos", min_value=0., max_value=1000000.,
                                                  value=df_final['amount'].mean(), step=100., help="Insira o valor médio dos empréstimos\
                                                  para calcular o impacto financeiro")
    if visualizacao == "Impacto Financeiro":
        taxa_juros = st.slider("Taxa média de juros", min_value=0, max_value=100, value=29,
                                help="Selecione a taxa de juros cobrada por empréstimo\
                                \ne veja como os valores se atualizam no gráfico") / 100            
    visualizar = st.button("Visualizar")

# Rodapé na barra lateral com as informações do desenvolvedor
    st.markdown(markdown, unsafe_allow_html=True)
           

if visualizar:    
    resultados, figura_impacto = calcular_e_plotar_impacto(matriz, valor_medio_emprest, taxa_juros)
    if visualizacao == "Análise Descritiva":        
        st.header("Análise Descritiva")
        st.markdown("<hr style='border: 2px solid #008000'>", unsafe_allow_html=True)
        col1, col2 = st.columns([0.55, 0.45], gap="medium")
        
        with col1:
            #fig_hist = plot_hist(df_final, df_final.select_dtypes(include=[np.number]).columns.tolist())
            #st.pyplot(fig_hist, use_container_width=True)
            st.write(valor_medio_emprest)
        
        with col2:
            st.markdown("<div style='font-size: 28px; font-weight: bold; color: #008000'>Relatório de Análise Descritiva", unsafe_allow_html=True)
            st.write("A análise descritiva dos dados é uma etapa fundamental para entender a distribuição e as características\
                      das variáveis numéricas do dataset. O histograma acima apresenta a distribuição de cada variável numérica,\
                      permitindo identificar padrões, tendências e possíveis outliers.")

    if visualizacao == "Impacto Financeiro": # Gráfico e relatório de impacto financeiro
        st.header("Impacto Financeiro")
        st.markdown("<hr style='border: 2px solid #008000'>", unsafe_allow_html=True)
        col1, col2 = st.columns([0.55,0.45], gap="medium")
        with col1:            
            st.pyplot(figura_impacto, use_container_width=True)               
            st.write(matriz)

        with col2:
            st.markdown("<div style='font-size: 28px; font-weight: bold; color: #008000'>Relatório de Impacto Detalhado", unsafe_allow_html=True)
            st.write(resultados.round(2))
            retorno_modelo = (resultados.iloc[1,3]) - (resultados.iloc[0,3])
            st.markdown(f"<div style='font-size: 23px; font-weight: bold; color: #008000'>Retorno\
                        líquido estimado utilizando o modelo: R$ {retorno_modelo:,.2f} ", unsafe_allow_html=True)
                        
            st.markdown("<hr style='border: 2px solid #008000'>", unsafe_allow_html=True)            
            st.markdown("<div style='font-size: 28px; font-weight: bold; color: #008000'>Descrição da visualização ", unsafe_allow_html=True)
            st.markdown("<div style='font-size: 18px; font-weight: sans serif'>O gráfico ao labo traz uma análise detalhada\
                        dos ganhos com bons pagadores, subtraindo-se as perdas com inadimplência e possível perda de clientes.\
                        Através dele é possível ter uma estimativa real dos possíveis\
                        retornos financeiros alcançáveis com o uso de Redes Neurais comparado a um modelo menos preciso e \
                        também ao cenário atual da empresa que não utiliza Inteligência Artificial na aprovação dos seus clientes (baseline).", unsafe_allow_html=True)                                            
     
    if visualizacao == "Redução da Inadimplência": # Gráfico e relatório de Inadimplência
        st.header("Redução da Inadimplência", anchor="red_inadimplencia")
        st.markdown("<hr style='border: 2px solid #2020df'>", unsafe_allow_html=True)
        col1, col2 = st.columns([0.6, 0.4], gap="small")
        with col1:            
            resultados = calcular_metricas(matriz)            
            figura_inad = plot_inadimplencia(resultados['inadimplencia_sem_modelos'],
                                         resultados['inadimplencia_prevista'])
            st.pyplot(figura_inad, use_container_width=True)
        
        with col2:
            st.markdown("<div style='font-size: 30px; font-weight: bold; color: #2020df'>Descrição da visualização", unsafe_allow_html=True)
            st.markdown("<div style='font-size: 20px; font-weight: bold'>Este gráfico compara três abordagens distintas e revela o poder dos modelos\
                         de inteligência artificial para transformar a gestão de risco e aumentar a rentabilidade nos negócios.", unsafe_allow_html=True)            
            
            st.markdown("<hr style='border: 2px solid #2020df'>", unsafe_allow_html=True) # Linha de separação

            st.markdown("<div style='font-size: 30px; font-weight: bold; color: #2020df'>Comparação de Cenários", unsafe_allow_html=True)
            st.markdown(f"<span style='font-size: 20px; font-weight: bold'>Taxa de Inadimplência Atual atinge alarmantes: \t </span>\
                      <span style='color: red; font-size: 25px; font-weight: bold'> {resultados['inadimplencia_sem_modelos']}%</span>", unsafe_allow_html=True)
            st.markdown(f"<span style='font-size: 20px; font-weight: bold'>Taxa de Inadimplência estimada com o uso do modelo mais\
                        eficaz (Redes Neurais) é de apenas: </span> <span style='color: green; font-size: 25px; font-weight: bold'>\
                        \t {resultados['inadimplencia_prevista']}0%</span>", unsafe_allow_html=True)
            
            st.markdown("<hr style='border: 2px solid #2020df'>", unsafe_allow_html=True) # Linha de separação

            st.markdown("<div style='font-size: 30px; font-weight: bold; color: #2020df'>Impacto final na Inadimplência", unsafe_allow_html=True)
            

    
    elif visualizacao == "Taxa de Aprovação": # Gráfico e relatório de taxa de aprovação
        st.header("Taxa de Aprovação")
        st.markdown("<hr style='border: 2px solid #2020df'>", unsafe_allow_html=True)
                
        st.write("Taxa de Aprovação do Modelo: ", resultado_xgb["taxa_aprovacao"])
        st.write("Taxa de Reprovação",  resultado_xgb['taxa_reprovacao'])
            
        
        st.markdown("<div style='font-size: 30px; font-weight: bold; color: #2020df'>Descrição da visualização", unsafe_allow_html=True)                      
        st.markdown("<div style=' font-size: 22px; font-weight:bold'>O gráfico ao lado mostra a taxa de aprovação \
                        de clientes. Note que Redes Neurais aprovou menos clientes que XGBoost, mas \
                        ainda assim conseguiu captar uma maior quantidade de bons clientes e consequentemente reprovou\
                        clientes com alto risco de se tornar inadimplentes, o que traz grandes benefícios, não\
                        só na maior captação de recursos mas também na prevenção de perdas financeiras.</div>", unsafe_allow_html=True )
        st.markdown("<hr style='border: 2px solid #2020df'>", unsafe_allow_html=True)
        st.markdown("<div style='font-size: 30px; font-weight: bold; color: #2020df'>Taxa de Aprovação dos Modelos", unsafe_allow_html=True)
                        
                       


