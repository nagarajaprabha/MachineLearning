# -*- coding: utf-8 -*-

import numpy as np,pandas as pd
import csv
from scipy.spatial import distance
import random
from numpy.random import permutation
from sklearn.neighbors import KNeighborsRegressor

trainingDataset = "";
testDataset = "";
global trainingData
testData="";
training_set=""
csvfile=""

def loadFile():
    with open('stats.csv') as csvfile:
        #training_set = csv.reader(csvfile, delimiter=',')
        #print(pd.read_csv(csvfile).columns.values) 
        trainingData = pd.DataFrame()
        trainingData=pd.read_csv(csvfile,header=1)
        # cleanup the columns
        trainingData.drop(trainingData.columns[[0,1,3,6,7,8]],inplace=True,axis=1)
        print(trainingData.shape)
    
#def predict():


loadFile()
#cleanData()
#predict()