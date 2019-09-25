'''   Clase
- Recibe las matrices
- Recibe el vocabulario (dict)
- dict{word, sim} findSimilar(V, threshold, year)
- lista histSimilar(V, threshold)
- array getVector(word, year)
- getVector(positives = [], negatives = [], year)
- getSim(w1, y1, w2, y2)
- getEvol(w1, y1, w2)
- list(year-1) getEvolByStep(w1)
'''

import numpy as np
from scipy import spatial
from os import listdir
from os.path import isfile, join
import pickle
import operator
import collections

'''
def validate(matrixes, vocabulary, yearDict):
    success = True
    if not isinstance(vocabulary, dict):
        success = False
    if not isinstance(yearDict, dict):
        success = False
    if matrixes
'''
    
class tempName:
    def __init__(self, model="dw2v", dataset="nyt"):
        with open("models/{}/{}/dictionary.pck".format(model,dataset),"rb") as f:
            dictionary = pickle.load(f)
        self.vocabulary = dictionary # vocabulario utilizado para generar las matrices
        self.inverseVocab = dict(map(reversed, dictionary.items())) # diccionario inverso del vocabulario
        mypath = "models/{}/{}/".format(model,dataset)
        onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
        self.cant_slices = len(onlyfiles)-1
        self.yearDict = {file.split("-")[1][:-4]:int(file.split("-")[0]) for file in onlyfiles if file != "dictionary.pck"}
        self.matrices = list(range(self.cant_slices))
        for file in onlyfiles:
            if file != "dictionary.pck":
                idx=int(file.split("-")[0])
                self.matrices[idx] = np.load("models/{}/{}/".format(model,dataset)+file)    # matrices de probabilidad de coocurrencia por year
        self.model = model
        self.dataset = dataset

    def findSimilars(self, vector, threshold, year):
        if threshold > 0:
            tempMat = self.matrices[self.yearDict[year]] # obtengo la matriz del anio pedido
            results = {} # container para los resultados encontrados dict{string, sim}
            # np.transpose(tempMat) si la palabra es la columna
            if type(threshold) is float:
                if threshold < 1.0:
                    for index, word in enumerate(tempMat): # ASUMIENDO QUE LA PALABRA ES LA FILA!!!
                        cosSim = 1 - spatial.distance.cosine(vector, word) # se obtiene la similitud coseno entre word y vector
                        if cosSim > threshold: # si la similitud coseno cae dentro del threshold dado
                            results[ self.inverseVocab[index] ] = cosSim # se guarda en results la palabra encontrada
                else:
                    raise ValueError("Similarity Threshold value must be under 1.0")
            elif type(threshold) is int:
                similarities = []
                for index, word in enumerate(tempMat): # ASUMIENDO QUE LA PALABRA ES LA FILA!!!
                    cosSim = 1 - spatial.distance.cosine(vector, word) # se obtiene la similitud coseno entre word y vector
                    similarities.append([index, cosSim]) # se guarda en similarities una lista de elementos [indice, similitud]
                similarities.sort(key=lambda elem: elem[1], reverse=True) # se ordena la lista de mayor a menor similitud
                mostSimilar = similarities[0:threshold] # se guardan en mostSimilar los 'threshold' elementos con mayor similitud
                for index, cosSim in mostSimilar:
                    results[ self.inverseVocab[index] ] = cosSim
            results = sorted(results.items(), key = operator.itemgetter(1), reverse = True) #orders the results
            return results

        else:
            raise ValueError("Treshold value must be positive")

    def histSimilar(self, vector, threshold):
        histograma = []

        return histograma

    def getVector(self, word, year):
        vector = self.matrices[self.yearDict[year]][self.vocabulary[word]]
        return vector

    def getVectorPosNeg(self, year, positives=[], negatives=[]):
        vector = np.zeros(self.matrices[self.yearDict[year]][0].shape)
        for word in positives:
            vector += self.matrices[self.yearDict[year]][self.vocabulary[word]]
        for word in negatives:
            vector -= self.matrices[self.yearDict[year]][self.vocabulary[word]]
        return vector

    def getSim(self, w1, y1, w2, y2):
        #return cosSim
        pass
    def getEvol(self, w1, y1, w2):
        #return evol
        pass
    def getEvolByStep(self, word):
        evolution = []
        return evolution
