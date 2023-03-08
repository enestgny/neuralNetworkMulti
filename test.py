from train import Definition
from train import Train
from train import neuralNetwork
import pandas as pd
import numpy as np


class test():
    def __init__(self,csv,weights):
        self.csv = csv
        self.weights = weights

    def start(self):
        m = 0
        w,w2,bias,bias2 = neuralNetwork('csv/train.csv').start()
        Test = Definition(self.csv)           #Ve sistemimizi tahmin yapabilir duruma getirmiş oluyoruz.
        CSV = Definition(self.csv)
        for k in range(len(Test.Species)):
            K1 = Train(w,bias,CSV.x[k],CSV.Species[k,4])
            aa = K1.ileriYayilim()
            a = K1.ActivationFuncLeakyRelu(aa)
            bbb = Train(w2,bias2,a,CSV.Species[k,4])
            bb = bbb.ileriYayilim()
            b = bbb.ActivationFuncLeakyRelu(bb)
            # print(CSV.Species[k,4],' value ',b)

            if   0 < b < 1.5:
                print(k+1,b,'Iris-setosa',Test.x_data[k,5])
                if Test.x_data[k,5] != 'Iris-setosa': #Hataların olduğunu ve kaç tane olduğunu çıktı olarak göstermesi
                    m += 1
                    print(f'Wrong {m}') 
            elif 1.5 < b < 2.3:
                print(k+1,b,'Iris-versicolor',Test.x_data[k,5])
                if Test.x_data[k,5] != 'Iris-versicolor':
                    m += 1
                    print(f'Wrong {m}') 
            elif 2.3 < b < 4:
                print(k+1,b,'Iris-virginica',Test.x_data[k,5])
                if Test.x_data[k,5] != 'Iris-virginica':
                    m+= 1
                    print(f'Wrong {m}') 

