from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from train import neuralNetwork
from test import test

df = pd.read_csv('Iris.csv')
column_name='Species'
targets=np.array(df[column_name])
df.drop(columns='Id',inplace=True)
x_type = np.array(df)

for i in range(len(targets)):
    if targets[i]=='Iris-setosa':
        targets[i] = 1
    elif targets[i] =='Iris-versicolor':
        targets[i] = 2
    elif targets[i] =='Iris-virginica':
        targets[i] = 3

x_train,x_test,y_train,y_test =  train_test_split(x_type,targets,test_size=0.2,shuffle=True)

pd.DataFrame(x_train).to_csv("csv/train.csv")
pd.DataFrame(x_test).to_csv("csv/test.csv")


test('csv/test.csv','weights.csv').start()
