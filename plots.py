        
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from sklearn.preprocessing import LabelEncoder
from joblib import dump

# ------------------------ Cálculo de Métricas e Impacto -----------------------------
def calcular_metricas(matriz_confusao):
    """
    Calcula métricas de desempenho do modelo com base na matriz de confusão.

    Parâmetros:
        matriz_confusao (np.array): Matriz de confusão no formato [[VN, FP], [FN, VP]]

    Retorna:
        dict: Dicionário com métricas de inadimplência, captação e aprovação
    """
    VN, FP = matriz_confusao[0]
    FN, VP = matriz_confusao[1]

    total = VN + FP + FN + VP
    total_aprovados = VN + FP
    total_reais_fraudulentos = FN + VP

    inadimplencia_real = (VP + FP) / total * 100 if total > 0 else 0
    inadimplencia_prevista = FP / total_aprovados * 100 if total_aprovados > 0 else 0
    captacao_bons_clientes = VP / total_reais_fraudulentos * 100 if total_reais_fraudulentos > 0 else 0
    taxa_aprovacao = (VP + FP) / total * 100 if total > 0 else 0
    taxa_reprovacao = 100 - taxa_aprovacao

    return {
        "inadimplencia_sem_modelos": round(inadimplencia_real, 2),
        "inadimplencia_prevista": round(inadimplencia_prevista, 2),
        "captacao_bons_clientes": round(captacao_bons_clientes, 2),
        "taxa_aprovacao": round(taxa_aprovacao, 2),
        "taxa_reprovacao": round(taxa_reprovacao, 2)
    }

# --------------------------------- Cálculo e Plot de Impacto Financeiro ------------------------

def calcular_e_plotar_impacto(matriz_confusao, valor_medio_emprestimo, taxa_juros):
    VN, FP = matriz_confusao[0][0], matriz_confusao[0][1]
    FN, VP = matriz_confusao[1][0], matriz_confusao[1][1]

    # Cálculo de valores financeiros
    ganho_bons = VN * valor_medio_emprestimo * taxa_juros
    perda_fraudes_aprovadas = FP * valor_medio_emprestimo
    perda_clientes_reprovados = FN * valor_medio_emprestimo * taxa_juros
    ganho_fraudes_reprovadas = VP * valor_medio_emprestimo

    # DataFrame para exibição
    df_impacto = pd.DataFrame({
        "Cenário": ["Ganho com bons clientes", "Perda por fraudes aprovadas", "Perda por bons reprovados", "Economia por fraudes reprovadas"],
        "Valor (R$)": [ganho_bons, -perda_fraudes_aprovadas, -perda_clientes_reprovados, ganho_fraudes_reprovadas]
    })

    # Gráfico com Plotly
    fig = go.Figure(go.Bar(
        x=df_impacto["Valor (R$)"],
        y=df_impacto["Cenário"],
        orientation='h',
        marker=dict(color=df_impacto["Valor (R$)"], colorscale='RdYlGn'),
        text=[f"R$ {v:,.2f}" for v in df_impacto["Valor (R$)"]],
        textposition="auto"
    ))

    fig.update_layout(
        title="💰 Impacto Financeiro do Modelo",
        xaxis_title="Valor Estimado (R$)",
        yaxis_title="Cenário",
        height=400,
        margin=dict(l=100, r=40, t=60, b=40)
    )

    return df_impacto, fig
# ----------------------- Plotagem das Inadimplências com e sem modelos -----------------------
def plot_inadimplencia(inadimplencia_sem_modelo, inadimplencia_com_modelo):
    """
    Gera gráfico de barras comparando a inadimplência com e sem o uso do modelo.

    Parâmetros:
        inadimplencia_sem_modelo (float): Taxa sem modelo
        inadimplencia_com_modelo (float): Taxa com modelo

    Retorna:
        matplotlib.figure.Figure: Figura pronta para exibição no Streamlit
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    modelos = ["Sem Modelo (Baseline)", "Com Modelo"]
    taxas = [inadimplencia_sem_modelo, inadimplencia_com_modelo]
    cores = ["#d9534f", "#5cb85c"]  # vermelho e verde

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=modelos, y=taxas, palette=cores, edgecolor="black", linewidth=1.5, ax=ax)

    for i, valor in enumerate(taxas):
        ax.text(i, valor + 0.5, f"{valor:.2f}%", ha="center", fontsize=12, fontweight="bold")

    ax.set_title("Comparação da Taxa de Inadimplência", fontsize=16)
    ax.set_ylabel("Taxa de Inadimplência (%)", fontsize=14)
    ax.set_xlabel("Cenário", fontsize=14)
    ax.set_ylim(0, max(taxas) + 10)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    return fig




def plot_hist(df, colunas_numericas):    
    sns.set_theme("poster", "darkgrid") 
    fig, ax = plt.subplots()
    
    for coluna in df[colunas_numericas]:        
                    
        sns.histplot(data=df, x = coluna, kde=True, ax=ax)
        ax.set_title(f"Histograma de {coluna}", fontsize=16, fontweight="bold")
        ax.set_ylabel(f"Frequência de {coluna}", fontsize=12, fontweight="bold")
        ax.set_xlabel(coluna, fontsize=12, fontweight="bold")
        sns.despine(right=True, top=True)            
        plt.tight_layout()
    return fig



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
}

melhores_parametros = {'boosting_type': 'gbdt', 'colsample_bytree': 0.8730817875436903, 'learning_rate': 0.07687319555336146, 'max_depth': 3, 'min_child_samples': 32, 'n_estimators': 324,
                        'num_leaves': 63, 'reg_alpha': 0.10660051374614865,
                        'reg_lambda': 0.009666508676232064, 'scale_pos_weight': 1.0231160617210455, 'subsample': 0.6496996236271484}

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
        """