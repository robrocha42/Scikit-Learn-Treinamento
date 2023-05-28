#Titanic - Kaggle
import pandas as pd
from sklearn.linear_model import LogisticRegression
import numpy as np

df = pd.read_csv('train.csv')

"""
Para treinar um modelo, precisamos separar as variáveis 
preditoras da variável resposta. Neste dataset, a variável 
resposta é a variável Survived que indica se o passageiro 
sobreviveu ou não. Todas as demais são variáveis preditoras
"""

"""
Por convenção, chama-se a matriz com as variáveis preditoras 
de X e o vetor com a variável resposta de y.
"""

#Separando as variáveis preditoras da variável resposta

X = df['Fare'].copy()
#formato adequado para serem consumidos pelo algoritmo
X = X.values.reshape(-1,1)
Y = df['Survived'].copy()


"""
Treinando um modelo de regressão logística 
utilizando o Scikit-Learn
"""
#instanciando
clf = LogisticRegression()
#Fit recebe as variáveis preditoras e a variável resposta
clf.fit(X,Y)
#retorna a acurácia do modelo - 66,5%
clf.score(X,Y)

#Utilizando o modelo treinado para fazer novas previsões
#O modelo prevê a classe 0 (não sobrevivência) para um passageiro cuja tarifa foi de 50 
#e prevê a classe 1 (sobrevivência) para um passageiro cuja tarifa foi de 70.
clf.predict(np.array([[50]]))

clf.predict(np.array([[70]]))

#prever a probabilidade de sobrevivência de um passageiro com base na tarifa paga
#a probabilidade de o passageiro sobreviver é de 45,5%, enquanto no segundo exemplo a probabilidade é de 53,1%
clf.predict_proba(np.array([[50]]))

clf.predict_proba(np.array([[70]]))