# -*- coding: utf-8 -*-

import numpy as np,pandas as pd
import csv
from scipy.spatial import distance
import random
from numpy.random import permutation
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import NearestNeighbors
import sqlite3
from pandas.io import sql
import datetime as dt
from sqlalchemy import create_engine # database connection
from pandas import DataFrame
trainingDataset = "";
testDataset = "";
global trainingData
testData="";
training_set=""
csvfile=""
out_sqlite = 'sqlite:///train.sqlite'
table_name = 'stats' # name for the SQLite database table
index_start = 1
disk_engine = create_engine(out_sqlite,echo=True)
df18 = "";
chunksize = 20000
j = 0
def load():
    index_start = 1
    start = dt.datetime.now()
    j = 1
    for df in pd.read_csv('stats.csv', chunksize=chunksize, iterator=True, encoding='utf-8'):

        df = df.rename(columns={c: c.replace(' ', '') for c in df.columns}) # Remove spaces from columns

        df.index += index_start


        j+=1
        print ("{} seconds: completed {} rows"+str((dt.datetime.now() - start).seconds)+" \t "+str( j*chunksize))

        df.to_sql('traindata', disk_engine, if_exists='append')
        index_start = df.index[-1] + 1
        #df = pd.read_sql_query('SELECT * FROM traindata LIMIT 3', disk_engine)
        df.head()

def loadFile():
    with open('stats.csv') as csvfile:
        #training_set = csv.reader(csvfile, delimiter=',')
        #print(pd.read_csv(csvfile).columns.values) 
        trainingData = pd.DataFrame(dtype=float)
        trainingData=pd.read_csv(csvfile,header=None)
        #print(trainingData.columns.values) 
        # cleanup the columns
        trainingData.drop(trainingData.columns[[0,3,6,7,8]],inplace=True,axis=1)
        
        grouped_df=trainingData.groupby([1,2])
        print(grouped_df.count())

        keys = grouped_df.groups.keys()
        #print(keys)

        #print(grouped.mean())
        #print(trainingData.shape)
        
    
def predict():
        index_start=1;
        innerit = 1
        outerit = 1
        query = '\
        select distinct player_fifa_api_id ,\
        player_api_id,\
        AVG(overall_rating),\
        AVG(potential),\
        AVG(crossing),\
        AVG(finishing),\
        AVG(heading_accuracy),\
        AVG(short_passing),\
        AVG(volleys),\
        AVG(dribbling),\
        AVG(curve),\
        AVG(free_kick_accuracy),\
        AVG(long_passing),\
        AVG(ball_control) ,\
        AVG(acceleration),\
        AVG(sprint_speed),\
        AVG(agility),\
        AVG(reactions),\
        AVG(balance),\
        AVG(shot_power),\
        AVG(jumping),\
        AVG(stamina),\
        AVG(strength),\
        AVG(long_shots),\
        AVG(aggression),\
        AVG(interceptions),\
        AVG(positioning),\
        AVG(vision),\
        AVG(penalties),\
        AVG(marking),\
        AVG(standing_tackle),\
        AVG(sliding_tackle),\
        AVG(gk_diving),\
        AVG(gk_handling),\
        AVG(gk_kicking),\
        AVG(gk_positioning),\
        AVG(gk_reflexes)\
        from traindata  ';
        
        trainQuery = query +\
        ' where player_api_id not in (30829,30962,30731)\
        group by player_fifa_api_id,player_api_id';
        print('Train Query')
        #testrfchunkdf =   pd.read_sql_query(testquery,disk_engine,chunksize=10000);
        exe  = disk_engine.execute(trainQuery)
        df1 = DataFrame(exe.fetchall())
        df1.columns = exe.keys()
        #TOTAL TEST RECORDS ARE 25laksh;2528242
        print('Train Query complete')
        print(df1);
        
        testQuery = query +\
        ' where player_api_id in (30829,30962,30731)\
        group by player_fifa_api_id,player_api_id';
        print('Test Query')
        exe  = disk_engine.execute(testQuery)
        df2 = DataFrame(exe.fetchall())
        df2.columns = exe.keys()
        #TOTAL TEST RECORDS ARE 25laksh;2528242
        print('Test Query complete')
        print(df2);
        # Create the knn model.
# Look at the five closest neighbors.
        knn = NearestNeighbors(n_neighbors=5, algorithm='ball_tree').fit(df1.fillna(0))
        # Fit the model on the training data.
        #knn.fit(trainQuery)
        # Make point predictions on the test set using the fit model.
        distances, indices = knn.kneighbors(df2.fillna(0))
        print(distances, indices)
        #for testchunkdf in testrfchunkdf:

predict()
#cleanData()
#predict()