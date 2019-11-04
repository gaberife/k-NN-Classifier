from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn import model_selection
from sklearn.metrics.scorer import make_scorer
import pandas as pd
from scipy.spatial import distance 
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

def readData(numRows = None):
    inputCols = ["Alcohol", "Malic Acid", "Ash", "Alcalinity of Ash", "Magnesium", "Total Phenols", "Flavanoids", "Nonflavanoid Phenols", "Proanthocyanins", "Color Intensity", "Hue", "Diluted", "Proline"]
    outputCol = 'Class'
    colNames = [outputCol] + inputCols  
    wineDF = pd.read_csv("data/wine.data", header=None, names=colNames, nrows = numRows)

    wineDF = wineDF.sample(frac=1, random_state=99).reset_index(drop=True)
    return wineDF, inputCols, outputCol 

class KNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, k = 1):
        self.k = k
        self.inputsDF = None
        self.outputSeries = None
        self.scorer = make_scorer(accuracyOfActualVsPredicted, greater_is_better=True)    
    def fit(self, inputsDF, outputSeries):
        self.inputsDF = inputsDF
        self.outputSeries = outputSeries
        return self
    def predict(self, testInput):
        if isinstance(testInput, pd.core.series.Series):
            return self.outputSeries.loc[findNearestWOK(self.inputsDF, testInput)]
        elif (self.k == 1):
            series = testInput.apply(lambda r: findNearestWOK(self.inputsDF, r),axis = 1)
            newSeries = series.map(lambda r: self.outputSeries.loc[r])
            return newSeries
        else:
            dataFrame = testInput.apply(lambda r: findNearestWithK(self.inputsDF, r, self.k),axis = 1)
            newDF = dataFrame.apply(lambda r: self.outputSeries.loc[r].mode())
            series = newDF.loc[:,0]
            return series
        
def accuracyOfActualVsPredicted(actualOutputSeries, predOutputSeries):
    compare = (actualOutputSeries == predOutputSeries).value_counts()
    if (True in compare):
        accuracy = compare[True] / actualOutputSeries.size
    else:
        accuracy = 0
    return accuracy

def findNearestWithK(df, testRow, k):
    nearestHOF = df.apply(lambda row: distance.euclidean(row, testRow),axis = 1)
    ksmall =  nearestHOF.nsmallest(k).index
    return ksmall
    
def findNearestWOK(df,testRow):
    nearestHOF = df.apply(lambda row: distance.euclidean(row, testRow),axis = 1)
    return nearestHOF.idxmin()  

def standardize(df, listCol):
    df.loc[:, listCol] = (df.loc[:, listCol] - df.loc[:, listCol].mean()) / df.loc[:, listCol].std()
    return df.loc[:, listCol]

def testKNN(k = 10):
    df, inputCols, outputCols = readData()
    alg = KNNClassifier()
    
    dfCopy = df.copy()
    results = model_selection.cross_val_score(alg, dfCopy.loc[:, inputCols], dfCopy.loc[:, outputCols], cv=k, scoring=alg.scorer)
    print("Unaltered Data 1-NN: ", results.mean() )
    
    dfCopy = df.copy()
    standardize(dfCopy, inputCols)
    results = model_selection.cross_val_score(alg, dfCopy.loc[:, inputCols], dfCopy.loc[:, outputCols], cv=k, scoring=alg.scorer)
    print("Standardized Data 1-NN: ", results.mean() )
    
    alg2 = KNNClassifier(8)
    dfCopy = df.copy()
    standardize(dfCopy, inputCols)
    results = model_selection.cross_val_score(alg2, dfCopy.loc[:, inputCols], dfCopy.loc[:, outputCols], cv=k, scoring=alg2.scorer)
    print("Standardized Data 8-NN: ", results.mean() )
    
    '''
    Answer Problem #2: K-1 overfits the dataset becasuse the nearest neighbor 
    to the training set is itself, if the dataset were smaller it wouldn't be
    as much of an issue
    '''
    
def plotKNN(k = 10):
    df, inputCols, outputCols = readData()
    standardize(df, inputCols)    
    neighborList = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 40, 50, 60, 80])
    accuracies = neighborList.apply(lambda r: model_selection.cross_val_score(
            KNNClassifier(r),df.loc[:, inputCols], df.loc[:, outputCols], 
            cv=k, scoring=KNNClassifier(r).scorer).mean())
    '''
    plt.plot(neighborList, accuracies)
    plt.xlabel('Neighbors')
    plt.ylabel('Accuracy')
    '''
    print(accuracies)
    print("Best: " , neighborList.loc[accuracies.idxmax()])
    
    
def builtInKNN():
    df, inputCols, outputCols = readData()
    standardize(df, inputCols) 
    
    neighborList = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 40, 50, 60, 80])
    dfCopy = df.copy()
    
    alg = KNeighborsClassifier(n_neighbors = 8)
    cvScores = model_selection.cross_val_score(alg, df.loc[:, inputCols], df.loc[:, outputCols], cv=10, scoring='accuracy')
    print("Standardized dataset, 8NN, accuracy: ", cvScores.mean())
    
    accuracies = neighborList.apply(lambda r: model_selection.cross_val_score(
            KNeighborsClassifier(r), dfCopy.loc[:, inputCols], dfCopy.loc[:, outputCols], 
            cv=10, scoring='accuracy').mean())
    print(accuracies) 
    
    plt.plot(neighborList, accuracies)
    plt.xlabel('Neighbors')
    plt.ylabel('Accuracy')
    
# -----------------------------------------------------------------------------
# Given
def test08():
    print("=====================","testKNN (using kNNClassifier class):", sep="\n", end="\n")
    testKNN()  # problem 2
    
    print("=====================","plotKNN (using kNNClassifier class):", sep="\n", end="\n")
    plotKNN()  # problem 3
    
    print("=====================","builtInKNN (using KNeighborsClassifier class):", sep="\n", end="\n")
    builtInKNN()  # problem 4

    
