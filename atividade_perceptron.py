"""
IA E APRENDIZAGEM DE MÁQUINA - REDES NEURAIS, ROBÓTICA E IOT

ATIVIDADE 1 - CLASSIFICAÇÃO DE FRUTAS UTILIZANDO RNAs - PERCEPTRON e
MULTI LAYER PERCEPTRON (MLP) - Scikit-Learn

DESENVOLVEDORES:
Fabiana Santos De Oliveira – RA 622100337
Hugo Leonardo Xavier - RA 622100640
João Vitor Fonseca da Silva – RA 622101634
Nathan Goggi – RA 622100654
Thiago Santos Amaral - RA 622100031
Vitor Renato Michelucci – RA 622101058

***Script adptado de implementações realizadas em aula***
"""
#%% Declarando as bibiliotecas e carregando o arwquivo de testes

import numpy as np
import sys

from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier

#definindo o caminho para o arquivo de treinamento
PATH=''

def processa_perceptron(padroes):
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
    print("Acurácia do Treinamento Perceptron = %.2f%%" %(acuracia_trein*100.0))

    def incializa_pesos(w):
        for i in range(len(w)):
            w[i] = np.random.uniform(-1, 1)
        return w

    w = np.zeros((num_atrib),'float')
    w = incializa_pesos(w)
    print('Saída utilizando a técnica de perceptron: ', clf.predict(padroes))

    #%%

    #Utilizando RNA MLP
    clf = MLPClassifier(hidden_layer_sizes=(7,), learning_rate='constant',learning_rate_init=0.007, max_iter=3000)

    #Treina a RNA
    clf.fit(X_train, y_train)

    #Calcula e imprime a acurácia do treinamento
    acuracia_trein = clf.score(X_train, y_train)
    print("Acurácia do Treinamento MLP = %.2f%%" %(acuracia_trein*100.0))

    #testes
    
    print('Saída utilizando a técnica de MLP: ', clf.predict(padroes))




print("Iniciando processamento do perceptron")

print("Exemplo 1:")
padroes_1 = np.array([[0.8, 0.5, 0.1],
                        [0.4, 0.5, 0.3],
                        [0.2, 0.2, 0.7]])
print(padroes_1)
processa_perceptron(padroes_1)

print("\nExemplo 2:")
padroes_2 = np.array([[0.2, 0.7, 0.3],
                        [0.10, 0.12, 0.3],
                        [0.6, 0.7, 0.8]])
print(padroes_2)
processa_perceptron(padroes_2)

print("\nExemplo 3:")
padroes_3 = np.array([[0.1, 0.4, 0.7],
                        [0.5, 0.7, 0.1],
                        [0.6, 0.9, 0.8],
                        [0.3, 0.7, 0.2]])
print(padroes_3)
processa_perceptron(padroes_3)

print("\nExemplo 4:")
padroes_4 = np.array([[1.1, 1.4, 1.7],
                    [4.5, 0.1, 0.5],
                    [0.5, 0.5, 0.5],
                    [0.0, 0.1, 5.2]])
print(padroes_4)
processa_perceptron(padroes_4)
