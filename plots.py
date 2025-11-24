        
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import LabelEncoder
from joblib import dump

# ------------------------ Cálculo de Métricas e Impacto -----------------------------

def calcular_metricas_fraude(matriz_confusao):
    """
    Calcula métricas de desempenho do modelo de detecção de fraudes com base na matriz de confusão.
    """
    VN, FP = matriz_confusao[0]
    FN, VP = matriz_confusao[1]

    total = VN + FP + FN + VP
    total_alertas = VP + FP
    total_fraudes_reais = FN + VP

    # Métricas principais
    precisao_alerta = VP / total_alertas * 100 if total_alertas > 0 else 0
    recall_fraude = VP / total_fraudes_reais * 100 if total_fraudes_reais > 0 else 0
    taxa_alerta = total_alertas / total * 100 if total > 0 else 0
    taxa_nao_alerta = 100 - taxa_alerta

    # Métricas adicionais
    f1_score = (2 * precisao_alerta * recall_fraude / (precisao_alerta + recall_fraude)
                if (precisao_alerta + recall_fraude) > 0 else 0)
    taxa_fp = FP / (FP + VN) * 100 if (FP + VN) > 0 else 0
    taxa_fn = FN / (FN + VP) * 100 if (FN + VP) > 0 else 0

    return {
        "precisao_alerta": round(precisao_alerta, 2),
        "recall_fraude": round(recall_fraude, 2),
        "taxa_alerta": round(taxa_alerta, 2),
        "taxa_nao_alerta": round(taxa_nao_alerta, 2),
        "f1_score": round(f1_score, 2),
        "taxa_falsos_positivos": round(taxa_fp, 2),
        "taxa_falsos_negativos": round(taxa_fn, 2)
    }

# --------------------------------- Impacto Financeiro ------------------------
def calcular_e_plotar_impacto(matriz_confusao, valor_medio_emprestimo, taxa_juros):
    VN, FP = matriz_confusao[0][0], matriz_confusao[0][1]
    FN, VP = matriz_confusao[1][0], matriz_confusao[1][1]

    # Cálculo de valores financeiros
    ganho_bons = VN * valor_medio_emprestimo * taxa_juros
    perda_fraudes_aprovadas = FP * valor_medio_emprestimo
    perda_clientes_reprovados = FN * valor_medio_emprestimo * taxa_juros
    ganho_fraudes_reprovadas = VP * valor_medio_emprestimo

    df_impacto = pd.DataFrame({
        "Cenário": ["Ganho com bons clientes", "Perda por fraudes aprovadas",
                    "Perda por bons reprovados", "Economia por fraudes reprovadas"],
        "Valor (R$)": [ganho_bons, -perda_fraudes_aprovadas,
                       -perda_clientes_reprovados, ganho_fraudes_reprovadas]
    })

    # Gráfico de barras interativo
    fig = px.bar(df_impacto, x="Cenário", y="Valor (R$)", color="Cenário",
                 text="Valor (R$)", title="💰 Impacto Financeiro do Modelo",
                 color_discrete_sequence=["#186826" if impacto > 0 else "#B11111" for impacto in df_impacto["Valor (R$)"]])
    fig.update_traces(texttemplate="R$ %{y:,.2f}", textposition="auto", textfont_size=14, 
                      )
    fig.update_layout(yaxis_range=[min(df_impacto["Valor (R$)"])* 12, max(df_impacto["Valor (R$)"])*1.1],
                      yaxis_zeroline=True, yaxis_zerolinecolor='grey', yaxis_zerolinewidth=1, 
                      legend=dict(orientation='v', yanchor='top', y=1.4, xanchor='center', x=0.85))

    return df_impacto, fig

# ----------------------- Comparação de Fraudes -----------------------
def plot_taxa_fraude(fraude_sem_modelo, fraude_com_modelo):
    """
    Gera gráfico de barras comparando a taxa de fraudes detectadas com e sem o uso do modelo.
    """
    df = pd.DataFrame({
        "Cenário": ["Sem Modelo (Baseline)", "Com Modelo"],
        "Taxa de Detecção (%)": [fraude_sem_modelo, fraude_com_modelo]
    })

    fig = px.bar(df, x="Cenário", y="Taxa de Detecção (%)", color="Cenário",
                 text="Taxa de Detecção (%)", title="Comparação da Taxa de Fraudes Detectadas")
    fig.update_traces(texttemplate="%{y:.2f}%", textposition="outside")

    return fig

# ----------------------- Gráficos adicionais -----------------------
def plot_proporcao_fraudes(matriz_confusao):
    """Pizza mostrando proporção de fraudes detectadas vs não detectadas"""
    FN, VP = matriz_confusao[1]
    df = pd.DataFrame({
        "Categoria": ["Fraudes Detectadas", "Fraudes Não Detectadas"],
        "Quantidade": [VP, FN]
    })
    fig = px.pie(df, values="Quantidade", names="Categoria",
                 title="Proporção de Fraudes Detectadas vs Não Detectadas",
                 color="Categoria", hole=0.4, color_discrete_sequence=['#186826', '#B11111'],
                 )
    fig.update_traces(textfont=dict(size=16),
                      textposition='inside',
                        hovertemplate='%{label}: %{value} (%{percent})<extra></extra>')
    
    
    return fig

def plot_radar_metricas(metricas):
    """Radar chart para comparar métricas de desempenho"""
    categorias = list(metricas.keys())
    valores = list(metricas.values())

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=valores,
        theta=categorias,
        fill='toself',
        name='Métricas',
        marker=dict(color='rgba(0, 255, 122, 0.7)',
                line=dict(color='rgba(20, 205, 62, 1)', width=5))
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100], dtick=20, color='grey')),
        title="Radar de Métricas de Desempenho"
    )
    return fig





# ------------------------- Pré-processamento de Colunas Categóricas ------------------------
def preprocessar_colunas_categoricas(X_treino, X_teste, colunas_categoricas):
    """
    Aplica Label Encoding nas colunas categóricas do DataFrame.
    
    Parâmetros:
    - df: DataFrame com os dados
    - colunas_categoricas: lista de nomes das colunas categóricas a serem processadas

    Retorna:
    - DataFrame com as colunas categóricas convertidas em numéricas
    """    
    
    for coluna in colunas_categoricas:
        le = LabelEncoder()
        X_treino[coluna] = le.fit_transform(X_treino[coluna].astype(str))
        X_teste[coluna] = le.transform(X_teste[coluna].astype(str))
        dump(le, f"objects/label_encoder_{coluna}.pkl")  # Salva o LabelEncoder para uso futuro
    
    return X_treino, X_teste

colunas_traduzidas = {
    'transaction_id': 'id_transacao',
    'user_id': 'id_usuario',
    'account_age_days': 'idade_conta_dias',
    'total_transactions_user': 'total_transacoes_usuario',
    'avg_amount_user': 'valor_medio_usuario',
    'amount': 'valor',
    'country': 'pais',
    'bin_country': 'pais_cartao',
    'channel': 'canal',
    'merchant_category': 'categoria_comerciante',
    'promo_used': 'promocao_usada',
    'avs_match': 'verificacao_endereco',
    'cvv_result': 'resultado_cvv',
    'three_ds_flag': 'autenticacao_3ds',
    'transaction_time': 'hora_transacao',
    'shipping_distance_km': 'distancia_envio_km',
    'is_fraud': 'fraude',
    'year': 'ano',
    'month': 'mes',
    'day': 'dia',
    'day_name': 'dia_semana',
    'is_weekend': 'fim_de_semana'
} # Dicionário para renomear colunas do dataset

melhores_parametros = {'boosting_type': 'dart',  'colsample_bytree': 0.7282628285096816, 'learning_rate': 0.2, 'max_depth': 14,
                        'min_child_samples':  74, 'n_estimators': 486, 'num_leaves': 20, 'reg_alpha': 1.948983261569964, 'reg_lambda': 10.0, 
                        'scale_pos_weight': 1.2097547688535297, 'subsample': 0.5} # Melhores parâmetros obtidos via Bayesian Optimization

markdown =  """
        <style>
        .footer {
        background-color: #f8f9fa;
        padding: 15px 20px;
        border-radius: 8px;
        text-align: center;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        margin-top: 40px;
        color: #343a40;
        }
        .footer a {
        margin: 0 15px;
        display: inline-block;
        }
        .footer img {
        height: 40px;
        width: auto;
        transition: transform 0.3s ease;
        }
        .footer img:hover {
        transform: scale(1.1);
        }
        </style>
        <div class="footer">
        <p><strong>Desenvolvido por: Ronivan</strong></p>
        <a href="https://github.com/Ronizorzan" target="_blank">
            <img src="https://img.icons8.com/ios-filled/50/000000/github.png" alt="GitHub">
        </a>
        <a href="https://www.linkedin.com/in/ronivan-zorzan-barbosa" target="_blank">
            <img src="https://img.icons8.com/color/48/000000/linkedin.png" alt="LinkedIn">
        </a>
        <a href="https://share.streamlit.io/user/ronizorzan" target="_blank">
            <img src="https://images.seeklogo.com/logo-png/44/1/streamlit-logo-png_seeklogo-441815.png" alt="Streamlit Community">
        </a>
        </div>
        """ # Rodapé com links para GitHub, LinkedIn e Streamlit Community