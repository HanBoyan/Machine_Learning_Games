import math
import random
import numpy as np
import matplotlib.pyplot as plt

def data_loader(fname:str):
    samples , labels= [],[]
    with open(fname, mode = 'r',encoding = 'UTF-8') as f:
        for data in f.readlines():
            samples.append([float(num) for num in data.split('\t')[:-1]])
            labels.extend([float(num) for num in data.split('\t')[-1][0]])
    for i in range(len(labels)):
        if labels[i] == 0.0:
            labels[i] = -1.0
    return samples,labels

def selectJrand(i,m):
    j = i
    while (j == i):
        j = int(random.uniform(0,m))
    return j

def clipAlpha(aj,H,L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj

def Simplified_SMO(samples,labels,C,toler,Max):
    samples,labels = np.mat(samples),np.mat(labels).transpose()
    b = 0
    m,n = samples.shape[0],samples.shape[1]
    alphas = np.mat(np.zeros((m,1)))
    ilte = 0
    while ilte < Max :
        alphaPairsChanged = 0
        for i in range(m):
            fXi = float(np.multiply(alphas,labels).transpose()*(samples*samples[i,:].transpose())) + b
            Ei = fXi - float(labels[i])#if checks if an example violates KKT conditions
            if ((labels[i]*Ei < -toler) and (alphas[i] < C)) or ((labels[i]*Ei > toler) and (alphas[i] > 0)):
                j = selectJrand(i,m)
                fXj = float(np.multiply(alphas,labels).transpose()*(samples*samples[j,:].transpose())) + b
                Ej = fXj - float(labels[j])
                alphaIold = alphas[i].copy(); alphaJold = alphas[j].copy();
                if (labels[i] != labels[j]):
                    L = max(0, alphas[j] - alphas[i])
                    H = min(C, C + alphas[j] - alphas[i])
                else:
                    L = max(0, alphas[j] + alphas[i] - C)
                    H = min(C, alphas[j] + alphas[i])
                if L==H: print( "L==H"); continue
                eta = 2.0 * samples[i,:]*samples[j,:].transpose() - samples[i,:]*samples[i,:].transpose() - samples[j,:]*samples[j,:].transpose()
                if eta >= 0: print ("eta>=0"); continue
                alphas[j] -= labels[j]*(Ei - Ej)/eta
                alphas[j] = clipAlpha(alphas[j],H,L)
                if (abs(alphas[j] - alphaJold) < 0.00001): print ("j not moving enough"); continue
                alphas[i] += labels[j]*labels[i]*(alphaJold - alphas[j])#update i by the same amount as j
                                                                        #the update is in the oppostie direction
                b1 = b - Ei- labels[i]*(alphas[i]-alphaIold)*samples[i,:]*samples[i,:].transpose() - labels[j]*(alphas[j]-alphaJold)*samples[i,:]*samples[j,:].transpose()
                b2 = b - Ej- labels[i]*(alphas[i]-alphaIold)*samples[i,:]*samples[j,:].transpose() - labels[j]*(alphas[j]-alphaJold)*samples[j,:]*samples[j,:].transpose()
                if (0 < alphas[i]) and (C > alphas[i]): b = b1
                elif (0 < alphas[j]) and (C > alphas[j]): b = b2
                else: b = (b1 + b2)/2.0
                alphaPairsChanged += 1
                print ("ilte: %d i:%d, pairs changed %d" % (ilte,i,alphaPairsChanged))
        if (alphaPairsChanged == 0): ilte += 1
        else: ilte = 0
        print ("iteration number: %d" % ilte)
    return b,alphas

def W_calculator(alphas,dataArr,labels):
    x,labels = np.mat(dataArr),np.mat(labels).transpose()
    m,n = x.shape[0],x.shape[1]
    w = np.zeros((n,1))
    for i in range(m):
        w += np.multiply(alphas[i]*labels[i],x[i,:].transpose())
    return w

def classifer(samples,w,b):
    samples = np.mat(samples)
    rst = []
    for i in range(len(samples)):
        pre = float(samples[i]*np.mat(w)+b)
        if pre < 0:
            pre = -1.0
        else:
            pre = 1.0
        rst.append(pre)
    return rst

if __name__ == '__main__':
    samples,labels = data_loader('testSet.txt')
    x_test,y_test,x_train,y_train = samples[80:],labels[80:],samples[:80],labels[:80]
    b,alphas = Simplified_SMO(x_train,y_train,0.5,0.1,500)
    w = W_calculator(alphas,x_train,y_train)
    rst = classifer(x_test,w,b)
    cnt = 0.0
    for i in range(len(rst)):
        if rst[i] == y_test[i]:
            cnt +=1
    print('AQC is',100*cnt/len(rst),'% in the end.')
    #'AQC is 95.0 % in the end.'