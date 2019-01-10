# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 15:57:10 2018

@author: s15106137
"""
VERBOSE = True
import warnings
import time
warnings.filterwarnings('ignore')

def visualise(X, y):
    import matplotlib.pyplot as plt
    # This is providing styles for plotting
    from matplotlib import style
    # Style specifier
    style.use('ggplot')
    
    #plt.legend(loc=4)
    #recall_score.plot()
    #precision_score.plot()
    X['0'].plot()
    y.plot()
    plt.xlabel('Schools')
    plt.ylabel('Score')
    plt.show()

#Prints a score to the console    
def printScore(score_array, label):
    import numpy as np
    avg = np.mean(score_array)
    std = np.std(score_array)
    #print(score_array)
    print("{:s}. Mean: {:f} - Standard Deviation: {:f}".format(label,avg,std))

#Evaluates model and gives f1-score. Generic to all algorithms
def scoreModel(model, X_test, y_test):
    from sklearn.metrics import f1_score
    from sklearn.metrics import precision_score
    from sklearn.metrics import recall_score
    y_pred = model.predict(X_test)
    from sklearn.metrics import classification_report
    #Global variable can be set to print a classification report, giving more information
    if(VERBOSE):
        print(classification_report(y_test,y_pred))
    return [f1_score(y_test, y_pred,average='weighted'), precision_score(y_test, y_pred,average='weighted'), recall_score(y_test, y_pred,average='weighted')]

#Performs kfold and records all the scores for each fold   
def foldData(param1, param2, param3, X,y):
    from sklearn.model_selection import StratifiedKFold
    kfold = StratifiedKFold(10, True, 1)
    startTime = 0
    endTime = 0
    f1_score_array = []
    precision_score_array = []
    recall_score_array = []
    time_taken_array = []
    #Print label
    print(str(param1) + "/" + str(param2) + "/" + str(param3))
    #Perform split
    for train, test in kfold.split(X, y):
        #Time is recorded before and after training the model
        startTime = time.time()
        model = trainModel(param1, param2, param3, X.iloc[train], y.iloc[train])
        endTime = time.time()
        #Score the model using the test data
        scores = scoreModel(model, X.iloc[test], y.iloc[test])
        #Record all the scores in their own arrays
        f1_score_array.append(scores[0])
        precision_score_array.append(scores[1])
        recall_score_array.append(scores[2])
        time_taken_array.append(endTime - startTime)
    #Use the score arrays to prduce averages and print them to console
    printScore(f1_score_array, "F1 Score       ")
    printScore(precision_score_array, "Precision Score")
    printScore(recall_score_array, "Recall Score   ")
    printScore(time_taken_array, "Time Taken     ")
    
#Method for training a specific model. Returns the model
def trainModel(param1, param2, param3, X_train, y_train): 
    from sklearn.neural_network import MLPClassifier
    if(param2 > 0):
        mlp = MLPClassifier(hidden_layer_sizes=(param1, param2, param3))
    else:
        mlp = MLPClassifier(hidden_layer_sizes=(param1,))
    return mlp.fit(X_train,y_train)

#Main method tying together the whole flow and defines file/column names
def ofsted():
    import pandas as pd
    #Remove index columns from files
    X = pd.read_csv('X.csv')
    X = X.drop("Unnamed: 0",axis=1)
    y = pd.read_csv('y.csv')
    y = y.drop("Unnamed: 0",axis=1)
    '''
    
    Uncomment lines to test other parameter sets
    
    foldData(10,0,0, X, y)
    foldData(20,0,0, X, y)
    foldData(24,0,0, X, y)
    foldData(30,0,0, X, y)
    foldData(40,0,0, X, y)
    foldData(50,0,0, X, y)
    foldData(75,0,0, X, y)
    foldData(100,0,0, X, y)
    foldData(10,10,10, X, y)
    foldData(20,20,20, X, y)
    foldData(24,24,24, X, y)
    foldData(30,30,30, X, y)
    foldData(40,40,40, X, y)
    foldData(50,50,50, X, y)
    foldData(75,75,75, X, y)
    foldData(100,100,100, X, y)
    foldData(300,0,0,X,y)
    foldData(500,0,0,X,y)
    foldData(1000,0,0,X,y)
    foldData(500,500,500,X,y)
    foldData(1000,1000,1000,X,y)'''
    foldData(300,300,300,X,y)
    
#Use preprocessing.py first to create processedData.csv
ofsted()

