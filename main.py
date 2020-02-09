import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from keras.models import Sequential
from keras.layers import Dense, Dropout

#Busca
csv = pd.read_csv('dados.csv', sep=';')

#Remove colunas desnecessárias
csv = csv.drop(columns=['Número comentários', 'Compartilhamento'])

#Trata
le = LabelEncoder()
csv['Tipo'] = le.fit_transform(csv['Tipo']) #Foto = 0|Link = 1|Status = 2|Video = 3
dados = csv.values

#Separa em Atributos e Classificadores
atributos = dados[:,:5]
classificadores = dados[:,5]

#Ajusta atributos de classificação não binária
ct = ColumnTransformer([('binario', OneHotEncoder(), [0, 1])], remainder='passthrough')
atributos = ct.fit_transform(atributos).toarray()

#Modelo
modelo = Sequential()
modelo.add(Dense(units=10, activation='relu'))
modelo.add(Dense(units=1, activation='linear'))
modelo.compile(loss = 'mean_absolute_error', optimizer = 'adam',
                  metrics = ['mean_absolute_error'])

modelo.fit(atributos, classificadores, batch_size=50, epochs=1000)

#Predizendo
#Tipo, Mes, Dia da Semana, Hora, Paga
novo = np.array([
    [0, 1, 7, 21, 1]
])
novo = ct.transform(novo).toarray()

retorno = modelo.predict(novo)
print('Média: ', int(retorno[0]))
