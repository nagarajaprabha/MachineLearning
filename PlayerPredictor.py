# -*- coding: utf-8 -*-

import numpy as np,pandas as pd
import csv
from scipy.spatial import distance
from numpy.random import permutation
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neighbors import NearestNeighbors
from sklearn.neighbors import KDTree
import sqlite3
from pandas.io import sql
import datetime as dt
from sqlalchemy import create_engine # database connection
from pandas import DataFrame
from sklearn import preprocessing
import matplotlib.pyplot as plt

from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from sklearn import cross_validation
from sklearn.linear_model import LinearRegression
import seaborn as sns

import plotly.plotly as py
from plotly.graph_objs import *
import plotly.tools as tls
import plotly.tools as tls

tls.set_credentials_file(username='PrabhaNagaraja', api_key='mzyhwxubbs')
py.sign_in("PrabhaNagaraja", "mzyhwxubbs")
trainingDataset = "";
testDataset = "";
global trainingData
testData="";
training_set=""
csvfile=""
out_sqlite = 'sqlite:///train4.sqlite'
table_name = 'stats' # name for the SQLite database table
index_start = 1
disk_engine = create_engine(out_sqlite,echo=False)
df18 = "";
chunksize = 20000
j = 0

#Graph properties
sns.set_style('darkgrid')

def load():
    index_start = 1
    start = dt.datetime.now()
    j = 1
    sqlite3.connect('train4.db').cursor().executescript('drop table if exists traindata;') 
    for df in pd.read_csv('stats - Copy.csv', chunksize=chunksize, iterator=True, encoding='utf-8'):

        df = df.rename(columns={c: c.replace(' ', '') for c in df.columns}) # Remove spaces from columns

        df.index += index_start
        label_encoder = preprocessing.LabelEncoder()
        #df['defensive_work_rate'] = df['defensive_work_rate'].factorize()[0]
        #df['defensive_work_rate']=label_encoder.fit_transform(df['defensive_work_rate'])
        
        df['attacking_work_rate'] = df['attacking_work_rate'].factorize()[0]
        df['attacking_work_rate']=label_encoder.fit_transform(df['attacking_work_rate'])
        
        df['preferred_foot'] = df['preferred_foot'].factorize()[0]
        df['preferred_foot']=label_encoder.fit_transform(df['preferred_foot'])
        
        #print(df['defensive_work_rate'])
        j+=1
        #print ("{} seconds: completed {} rows"+str((dt.datetime.now() - start).seconds)+" \t "+str( j*chunksize))

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
        select  \
        player_api_id,\
        AVG(preferred_foot),\
        AVG(attacking_work_rate),\
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
        group by player_api_id';
        print('Train Query')
        #testrfchunkdf =   pd.read_sql_query(testquery,disk_engine,chunksize=10000);
        exe  = disk_engine.execute(trainQuery)
        df1 = DataFrame(exe.fetchall())
        df1.columns = exe.keys()
        #df3 = df1.copy()
        #df3.drop(df3.columns[[0]],inplace=True,axis=1)
        df3=df1.ix[:,1:35]
       
        #TOTAL TEST RECORDS ARE 25laksh;2528242
        print('Train Query complete')
        #print(df1);
        
        testQuery = query +\
        ' where player_api_id in (30829,30962,30731)\
        group by player_api_id';
        print('Test Query')
        exe  = disk_engine.execute(testQuery)
        df2 = DataFrame(exe.fetchall())
        df2.columns = exe.keys()
        df4=df2.ix[:,1:35]
        
        print('Test Query complete')
        #print(df2);
        # Create the knn model.
        # Look at the five closest neighbors.
        knn = NearestNeighbors(n_neighbors=5, algorithm='kd_tree').fit(df3.fillna(0))
        distances, indices = knn.kneighbors(df4.fillna(0))
        #print(indices);
        #print(distances)
        #for s in np.nditer(indices):
         #   print(s)
        colors = {'Andrea Pirlo': 'rgb(31, 119, 180)', 
          'Rooney': 'rgb(255, 127, 14)', 
          'Ramos': 'rgb(44, 160, 44)'}
        playerDF_collection = {} 
        traces = []
        legend = {0:False, 1:False, 2:False, 3:True}
        i = 0
        for s in indices:
            pca = PCA(n_components=5)
            #print(np.array(s).tolist())
            l = np.array(s).tolist();
            #playerDF_collection[i] = pd.DataFrame(df1.ix[l,:35]);
            testDf = pd.DataFrame(df2.ix[i,:35])
            testDf = testDf.transpose()
            trainDF= pd.DataFrame(df1.ix[l,:35])
            trainDF.reset_index();
            #print(trainDF)
            result = [testDf,trainDF]
            playerDF_collection[i] =  pd.concat(result)
            i=i+1;
            #print(playerDF_collection.dType)
            #playerDF_collection is dict type
            k = 0
        for j in playerDF_collection:
            X = playerDF_collection[j].iloc[:,0]
            Y = playerDF_collection[j].iloc[:,1:5]
            Y = Y.reset_index()
            #print(Y.type)
            #print(Y.loc[4198]);
            k = k+1
            for col in range(4):
                 for key in colors:
                     traces.append(Histogram(y=Y.loc[k], 
                        opacity=0.75,
                        xaxis='x%s' %(col+1),
                        marker=Marker(color=colors[key]),
                        name=key,
                        showlegend=legend[col]))
                    
            data = Data(traces)
            layout = Layout(barmode='overlay',
                xaxis=XAxis(domain=[0, 0.25], title='sepal length (cm)'),
                xaxis2=XAxis(domain=[0.3, 0.5], title='sepal width (cm)'),
                xaxis3=XAxis(domain=[0.55, 0.75], title='petal length (cm)'),
                xaxis4=XAxis(domain=[0.8, 1], title='petal width (cm)'),
                yaxis=YAxis(title='count'),
                title='Distribution of the different Iris flower features')

            fig = Figure(data=data, layout=layout)
            py.iplot(fig)
            Y_reduced = pca.fit_transform(scale(Y))
            #print(Y_reduced)
            #pca.fit(Y_reduced)
            var1 = np.cumsum(np.round(pca.explained_variance_ratio_, decimals=4)*100)
            plt.plot(var1)
            #print(var1)
            
            
            
            
                
            #print(playerDF_collection[j])
            #print(j.type)
            #X = playerDF_collection[j]
            #Y = playerDF_collection[j].iloc[:,1:35]
            #print(X)
            
            #Continue to plot the graph
        #print(playerDF_collection[1]);
                
            #for l in np.array(s).tolist():
            #    print(df1.loc[[l],:])
            #df1.loc[[np.array(s)],:]
        #kdt = KDTree(df1.fillna(0), leaf_size=30)
        #distances, indices=kdt.query(df2.fillna(0), k=5, return_distance=False)
        # Fit the model on the training data.
        # Make point predictions on the test set using the fit model.
        
        #print( indices)
        #print(df1.get(indices));
        #list(df1.index.values);
        #print(df1.shape);
        #print(df2.shape)
        #print(df1.loc[[4198,3116,1861,837,5467],:])
        #print(df1['player_api_id'])
        #print(df1.loc[df1['player_api_id'] == 203396].index)
        #print(df1.loc[8068])
        #print(df2.loc[df2['player_api_id'] == 30829])
        #print(df2)
        
        #print(df1.ix[1895])
        #for testchunkdf in testrfchunkdf:
#For first call uncomment load() method to run the dataload;for subsequent run comment out
#load()
predict()
#cleanData()
#predict()