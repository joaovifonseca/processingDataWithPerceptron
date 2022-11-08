"""
IA E APRENDIZAGEM DE MÁQUINA - REDES NEURAIS, ROBÓTICA E IOT

ATIVIDADE 1 - CLASSIFICAÇÃO DE FRUTAS UTILIZANDO RNAs - PERCEPTRON e
MULTI LAYER PERCEPTRON (MLP) - Scikit-Learn

DESENVOLVEDORES:
HUGO LEONARDO XAVIER
JOÃO VITOR FONSECA
VITOR RENATO MICHELUCCI

***Script adptado de implementações realizadas em aula***
"""
#%% Declarando as bibiliotecas e carregando o arwquivo de testes

import numpy as np
import sys

from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier

#definindo o caminho para o arquivo de treinamento
PATH=''

#lê o conjunto de treinamento
ct = np.loadtxt(PATH+'Frutas_Tipos.txt')
num_padroes, num_atrib = ct.shape

X_train = ct[:,0:num_atrib-1]
y_train = ct[:,num_atrib-1]

#Teste da leitura
print('Arquivo lido\n', ct)

#%%

#Utilizando RNA Perceptron
clf = Perceptron(tol=1e-5, random_state=0)

#Treina a RNA
clf.fit(X_train, y_train)

#Calcula e imprime a acurácia do treinamento
acuracia_trein = clf.score(X_train, y_train)
print("Acurácia do Treinamento = %.2f%%" %(acuracia_trein*100.0))

#testes

def incializa_pesos(w):
    for i in range(len(w)):
        w[i] = np.random.uniform(-1, 1)
    return w

w = np.zeros((num_atrib),'float')
w = incializa_pesos(w)
padroes = np.array([[0.7, 0.2, 0.5],
                    [0.9, 0.6, 0.3],
                    [0.2, 0.5, 0.9]])
print('As saídas para o conjunto de padrões é: ', clf.predict(padroes))

#%%

#Utilizando RNA MLP
clf = MLPClassifier(hidden_layer_sizes=(10,), learning_rate='constant',learning_rate_init=0.005, max_iter=3000)

#Treina a RNA
clf.fit(X_train, y_train)

#Calcula e imprime a acurácia do treinamento
acuracia_trein = clf.score(X_train, y_train)
print("Acurácia do Treinamento = %.2f%%" %(acuracia_trein*100.0))

#testes
padroes = np.array([[0.7, 0.2, 0.5],
                    [0.9, 0.6, 0.3],
                    [0.2, 0.5, 0.9]])
print('As saídas para o conjunto de padrões é: ', clf.predict(padroes))


