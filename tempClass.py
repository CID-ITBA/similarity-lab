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
    def __init__(self, matrixes, yearDict, vocabulary):
        self.vocabulary = vocabulary # vocabulario utilizado para generar las matrices
        self.inverseVocab = dict(map(reversed, vocabulary.items())) # diccionario inverso del vocabulario
        print(self.inverseVocab)
        self.matrixes = matrixes    # matrices de probabilidad de coocurrencia por year
        self.yearDict = yearDict    # diccionario que vincula year con indice en matrixes

    def findSimilars(self, vector, threshold, year):
        if threshold > 0:
            tempMat = self.matrixes[self.yearDict[year]] # obtengo la matriz del anio pedido
            results = {} # container para los resultados encontrados dict{string, sim}
            # np.transpose(tempMat) si la palabra es la columna
            if type(threshold) is float:
                if threshold < 1.0:
                    for index, word in enumerate(tempMat): # ASUMIENDO QUE LA PALABRA ES LA FILA!!!
                        cosSim = 1 - spatial.distance.cosine(vector, word) # se obtiene la similitud coseno entre word y vector
                        if np.abs(cosSim) > threshold: # si la similitud coseno cae dentro del threshold dado
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
            return results

        else:
            raise ValueError("Treshold value must be positive")

    def histSimilar(self, vector, threshold):
        histograma = []

        return histograma

    def getVector(self, word, year):
        vector = []

        return vector

    def getVectorPosNeg(self, positives, negatives, year):
        vector = []

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

ma = [[-1,-2,-3],[4,5,6],[7,8,9]]
mb = [[1,2.1,3],[4.2,4.8,6],[7.02,8,9.3]]
mc = [[1.1,2.2,3.1],[4.23,5,6],[7.03,8,9.32]]

matrixes = [ma, mb, mc]
yearDict = {1990:0, 1991:1, 1995:2}
vocabulary = {'martin':0, 'pablo':1, 'carlos':2}

tempObject = tempName(matrixes, yearDict, vocabulary)

newVec = tempObject.findSimilars([1,2,3], 3, 1990)

print(newVec)