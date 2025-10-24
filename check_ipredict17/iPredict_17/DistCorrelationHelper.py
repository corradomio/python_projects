import pandas as pd
import numpy as np
from scipy.spatial.distance import pdist, squareform



def distance_correlation(X, Y):
        distanceMatrixA = calculateDistanceMatrix(X)
        distanceMatrixB = calculateDistanceMatrix(Y)
        A = calculateDoublyCenteredDistanceMatrix(distanceMatrixA)
        B = calculateDoublyCenteredDistanceMatrix(distanceMatrixB)
        return calculateDistanceCorrelation(A, B)
        
def calculateDistanceMatrix(values):
            distanceMatrix = list([])
            for i in range(len(values)):
                row = []
                for j in range(len(values)):
                    euclidian_norm = np.abs(values[i] - values[j])
                    row.append(euclidian_norm)
                distanceMatrix.append(row)
            return distanceMatrix

def calculateDoublyCenteredDistanceMatrix(matrix):
        doublyCenteredDistanceMatrix = list([])
        rowMeans = []
        columnMeans = []
        grandMean = 0.0;
        #first calculate row, column and grand means of distance matrix
        for i in range(len(matrix)):
            rowTotal = 0.0
            columnTotal = 0.0
            for j in range(len(matrix)):
                rowTotal = rowTotal + matrix[i][j]
                grandMean = grandMean + matrix[i][j]
                columnTotal = columnTotal + matrix[j][i]
            rowMeans.append(rowTotal / len(matrix))
            columnMeans.append(columnTotal / len(matrix))
        grandMean = grandMean / (len(matrix) * len(matrix))
        #now build doubly centered distance matrix
        for i in range(len(matrix)):
            row = []
            for j in range(len(matrix)):
                current_value = matrix[i][j] - rowMeans[i] - columnMeans[j] + grandMean
                row.append(current_value)
            doublyCenteredDistanceMatrix.append(row)
        return doublyCenteredDistanceMatrix

def dCovXY(A, B):
        totalProduct = 0.0;
        for i in range(len(A)):
            for j in range(len(A)):
                totalProduct = totalProduct + A[i][j] * B[i][j]
        current_value = totalProduct / (len(A) * len(A))
        return np.sqrt(current_value)


def calculateDistanceCorrelation(A, B):
        distanceCorrelation = 0.0
        distanceCovXY = dCovXY(A, B)
        distanceStdDevX = dCovXY(A, A)
        distanceStdDevY = dCovXY(B, B)
        if (not(distanceStdDevX > 0) or not(distanceStdDevY > 0)):
            distanceCorrelation = 0.0
        else:
            distanceCorrelation = distanceCovXY / np.sqrt(distanceStdDevX * distanceStdDevY)
        return distanceCorrelation