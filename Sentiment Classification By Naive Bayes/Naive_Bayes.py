import numpy as np
import random as rd
import math
import re
import matplotlib.pyplot as plt





def data_load(file1:str,file2:str):
    #file2 is the label one in the project
    data =[]
    with open(file = file1,mode = 'r',encoding = 'utf-8') as f:
        data1 = []
        for sen in [inst.strip().split('\t') for inst in f.readlines()][1:]:
            data1.append([sen[1]])
       
    with open(file = file2,mode = 'r',encoding = 'utf-8') as f:
        label = []
        for sen in [inst.strip().split('\t') for inst in f.readlines()][1:]:
            label.append(int(str(sen)[-3]))
        
    if len(data1) == len(label):
        for i in range(len(label)):
            data1[i].append(label[i])
            data.append(data1[i])
        rd.shuffle(data)#数据集的标签集中性过于突出，随机打乱提高了约7个百分点，均匀分布数据效果？
        return data
    else:
        print('Num of samples is not equal to labels.')





def data_clean(data:list):
    #deal with "'s" and '.'
    new_data = []
    for txts in data:
        txt = txts[0].split(' ')
        cnt = 0
        for i in range(len(txt)):
            if re.match('\w+',txt[i]) == None:
                txt[i] = ' '
                cnt+=1
            if txt[i] == "'s":
                txt[i-1] += txt[i]
                txt[i] = ' '
                cnt+=1
        for i in range(cnt):
            txt.remove(' ')
        txt.append(txts[1])
        new_data.append(txt)
    return new_data





def VocabList(data:list):
    vocabset= set([])
    for text in data:
        vocabset = vocabset | set(text[:])
    return list(vocabset)





def word2vec(vocablist,data):
    '''
       The input data should be a vector like:['It','takes', 'a', 'really','long', 'slow', 'and', 'dreary', 'time', 'to', 'dope', 'out', 
       'what','TUCK','EVERLASTING','is','about',1]
    '''
    word_vec = [0] * len(vocablist)
    for i in range(len(data)-1):
        if data[i] in vocablist:
            word_vec[vocablist.index(data[i])] += 1
        else:print(data[i],'is not in the VocabBase!')
    return word_vec





def NBtrainer(word_vecs,labels):
    num_of_samples = len(word_vecs)
    num_of_words = len(word_vecs[0])
    tp1,tp2,tp3 = 0,0,0
    for tp in labels:
        if tp == 1:
            tp1+=1
        if tp == 2:
            tp2+=1
        if tp == 3:
            tp3+=1
    p1,p2,p3 = tp1/len(labels),tp2/len(labels),tp3/len(labels)
    p1cnt,p2cnt,p3cnt = np.ones(num_of_words),np.ones(num_of_words),np.ones(num_of_words)
    p1denom,p2denom,p3denom = 2.0,2.0,2.0
    for i in range(num_of_samples):
        if labels[i] == 1:
            p1cnt += word_vecs[i]
            p1denom += sum(word_vecs[i])
        elif labels[i] == 2:
            p2cnt += word_vecs[i]
            p2denom += sum(word_vecs[i])
        else:
            p3cnt += word_vecs[i]
            p3denom += sum(word_vecs[i])
    p1vec = np.log(p1cnt/p1denom)
    p2vec = np.log(p2cnt/p2denom)
    p3vec = np.log(p3cnt/p3denom)
    return p1vec,p2vec,p3vec,p1,p2,p3





def NBclassifer(vec2classify,p1vec,p2vec,p3vec,p1,p2,p3):
    prob1 = sum(vec2classify * p1vec) + math.log(p1)
    prob2 = sum(vec2classify * p2vec) + math.log(p2)
    prob3 = sum(vec2classify * p3vec) + math.log(p3)
    if max(prob1,prob2,prob3) == prob1:
        return 1
    elif max(prob1,prob2,prob3) == prob2:
        return 2
    elif max(prob1,prob2,prob3) == prob3:
        return 3





if __name__ == '__main__':  
    data = data_clean(data_load('datasetSentences.txt','datasetSplit.txt'))
    data_train,data_test = data[0:9439],data[9440:11799]
    vocablist = VocabList(data)
    train_labels = [l[-1] for l in data_train]
    test_labels = [l[-1] for l in data_test]
    train_word_vectors,test_word_vectors = [],[]
    for dat in data_train:
        train_word_vectors.append(word2vec(vocablist,dat))
    for dat in data_test:
        test_word_vectors.append(word2vec(vocablist,dat))
    trainer = NBtrainer(train_word_vectors,train_labels)
    print(trainer)
   
    crt = 0.0
    cnt = 0
    for i in range(len(test_word_vectors)):
        rst = NBclassifer(test_word_vectors[i],trainer[0],trainer[1],trainer[2],trainer[3],trainer[4],trainer[5])
        if rst == test_labels[i]:
            crt+=1
        cnt+=1
        acrcy = float(crt/cnt)
        print('Sample',i,'of type',test_labels[i],'is prediceted as type',rst,'.Accuracy is',acrcy,'now.')







