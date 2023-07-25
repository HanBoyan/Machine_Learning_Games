


import matplotlib.pyplot as plt
import numpy as np
import math




def data_loader(fname:str):
    with open(file = fname,mode = 'r',encoding = 'UTF-8') as f:
        data =[]
        for line in f.readlines():
            data.append(line.split('\t'))
        for sample in data:
            for i in range(len(sample)):
                sample[i] = float(sample[i])
        return data




def Sigmond(x):
    return 1.0/(1+np.exp(-x))




#def gradAscent(data_matrix,label):   批处理梯度上升
def stocGradAscent(data_matrix,label):
    data_matrix = np.mat(data_matrix)
    label = np.mat(label).transpose()
    m,n = data_matrix.shape[0],data_matrix.shape[1]
    alpha = 0.001
    #maxCycles = 500
    weights = np.ones((n,1))
    #for k in range(maxCycles):
    for i in range(m):
        h = Sigmond(sum(data_matrix[i] * weights))
        error = (label[i] - h)
        weights += alpha*data_matrix[i].transpose()*error #Logistic回归之梯度上升算法 Author:7yeers
    return weights





def Classifer(x):
    if x >=0.5:
        return 1
    else:
        return 0




if __name__ == '__main__'
    train,test = data_loader('horseColicTraining.txt'),data_loader('horseColicTest.txt')
    train_x,train_y,test_x,test_y = [],[],[],[]
    for sample in train:
        train_x.append(sample[:-1])
        train_y.append(sample[-1])
    weight = []
    for w in stocGradAscent(train_x,train_y):
        weight.append(w[0])
    for sample in test:
        test_x.append(sample[:-1])
        test_y.append(sample[-1])

    pred = []

    for data in test_x:
        rst =Classifer(Sigmond(np.mat(data) * np.mat(weight).transpose()))
        pred.append(rst)
    true = 0.0
    for i in range(len(test_x)):
        if pred[i] == test_y[i]:
            true+=1
    print('AQC is:',(true * 100)/len(test_x),'%.')
    #AQC is: 52.23880597014925 %.
