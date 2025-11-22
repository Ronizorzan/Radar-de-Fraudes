import pandas as pd
import numpy as np
from joblib import dump
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, chi2
from lightgbm import LGBMClassifier
from plots import preprocessar_colunas_categoricas, melhores_parametros, colunas_traduzidas

# Carregar e preparar os dados
np.random.seed(1432)
dataset = pd.read_csv("Fraud_transactions.csv")
dataset.rename(columns=colunas_traduzidas, inplace=True) # Renomeia com dicionário de plots.py

# Mapear valores categóricos
mapeamentos = {0: 'Não', 1: 'Sim'}
dataset['autenticacao_3ds'] = dataset['autenticacao_3ds'].map(mapeamentos)
dataset['promocao_usada'] = dataset['promocao_usada'].map(mapeamentos)
dataset['resultado_cvv'] = dataset['resultado_cvv'].map(mapeamentos)
dataset['verificacao_endereco'] = dataset['verificacao_endereco'].map(mapeamentos)  

# Separar variáveis
X = dataset.drop(['fraude', 'id_transacao', 'id_usuario', 'hora_transacao'], axis=1)
y = dataset['fraude']


# Codificar variáveis categóricas
categoricas = X.select_dtypes(include='object').columns.tolist()
X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.25, random_state=1432)
X_treino, X_teste = preprocessar_colunas_categoricas(X_treino, X_teste, categoricas) # Acessa função do plots.py

# Seleção de características
seletor = SelectKBest(chi2, k=10)
X_treino_final = seletor.fit_transform(X_treino, y_treino)
X_teste_final = seletor.transform(X_teste)
colunas_selecionadas = X_treino.columns[seletor.get_support()].tolist()

# Treinar modelo
lgbm_model = LGBMClassifier(**melhores_parametros) # Melhores parâmetros encontrados com Bayesian Optimization
final_model = lgbm_model.fit(X_treino_final, y_treino)

# Avaliar
y_pred = final_model.predict_proba(X_teste_final)[:, 1]
y_pred_bin = (y_pred >= 0.3).astype(int)
accuracy = accuracy_score(y_teste, y_pred_bin)
confusion = confusion_matrix(y_teste, y_pred_bin)

print(f"Acurácia do modelo: {accuracy*100:.2f}%")
print("Matriz de Confusão:")
print(confusion)
print(colunas_selecionadas)

# Salvar modelo e objetos
dump(final_model, "objects/modelo_fraude.pkl")
dump(colunas_selecionadas, "objects/colunas_selecionadas.pkl")
dump(seletor, "objects/seletor.pkl")
dump((accuracy, confusion), "objects/metricas.pkl")

