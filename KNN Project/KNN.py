import tensorflow as tf
import numpy as np
import math


def Distance(arr1,arr2):
    if len(arr1) != len(arr2):
        return None
    else:
        tmp = 0
        for i in range(0,len(arr1)):
            tmp += math.pow((arr1[i]-arr2[i]),2)
        return math.sqrt(tmp)


def Classifer(sample, target):
    dis = []
    for i in range(0, len(sample)):
        poi = i
        dis.append([Distance(sample[i], target), poi])

    def sort(a):
        for i in range(0, len(a)):
            for j in range(0, len(a) - i - 1):
                tmp = 0
                if a[j][0] >= a[j + 1][0]:
                    tmp = a[j]
                    a[j] = a[j + 1]
                    a[j + 1] = tmp
        return a

    dis = sort(dis)
    return dis[0:30]


minist = tf.keras.datasets.mnist
(train_x,train_y),(test_x,test_y) = minist.load_data()
data = []
for i in range(0,len(train_x)):
    temp=[]
    for j in train_x[i]:
        temp = np.append(temp,j)
    data.append(temp)
data = np.array(data)
sample_x,sample_y,test_x,test_y= data[0:9999],train_y[0:9999],data[10000:11999],train_y[10000:11999]
true = 0
false = 0
for i in range(0,len(test_x)):
    s = Classifer(sample_x,test_x[i])
    count = [0,0,0,0,0,0,0,0,0,0]
    for num in s:
        count[sample_y[num[1]]] += 1
    predict = count.index(max(count))
    print('Prediction as',predict,'while true number is',test_y[i])
    if predict == test_y[i]:
        true += 1
    else:
        false +=1
    rate = true*100/(true+false)
    print('Rate of true is',rate,'%')