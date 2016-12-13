import numpy as np
from numpy import linalg
import math
import csv

def get_x(line):
    ## Add bias to x at index 0
    newLine = [1.0]
    for i in range(1, len(line)):
        newLine.append(float(line[i]))
    return newLine

def MatricesXY(Data):
    X = []
    Y = []
    for row in Data:
        x= get_x(row)
        X.append(x)
        y = float(row[0])
        Y.append(y)
    return np.array(X), np.array(Y)

##------- Compute the pseudo inverse of X -------##
def PseudoInverse(X):
    X_T    = np.array(X.transpose())
    M      = np.dot(X_T, X)
    print M
    M_inv  = linalg.inv(M)
    X_pinv = np.dot(M_inv, X_T)
    return X_pinv
##-----------------------------------------------##

def LinearRegression(X, Y):#Data):
   #X, Y   = MatricesXY(Data)
   X_pinv = linalg.pinv(X)
   #PseudoInverse(X)
   return np.dot(X_pinv, Y)
