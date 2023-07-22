# %%
import math
import matplotlib.pyplot as plt
import operator



def Entropy_cal(data):
    quantities = len(data)
    dic = {}
    for features in data:
        if features[-1] not in dic.keys():
            dic[features[-1]] = 0
        dic[features[-1]] += 1
    Entropy = 0.0
    for value in dic.values():
        prob = float(value) / quantities
        Entropy -= prob * math.log(prob, 2)
    return Entropy



def Split_Data(data: list, axis, value):
    newdata = []
    for features in data:
        if features[axis] == value:
            tmp = features[:axis]
            tmp.extend(features[axis + 1:])
            newdata.append(tmp)
    return newdata



def BestSplitingWay(data):
    num_of_ways = len(data[0])
    init_entropy = Entropy_cal(data)
    entropy_dict = {}
    for features in data[0]:
        feavec = [fea[data[0].index(features)] for fea in data]
        kinds = set(feavec)
        for value in kinds:
            new_data = Split_Data(data, data[0].index(features), value)
            new_entropy = Entropy_cal(new_data)
            if features not in entropy_dict.keys():
                entropy_dict[features] = init_entropy - new_entropy
    best_feature = list(entropy_dict.values()).index(min(entropy_dict.values()))
    return best_feature



def Most_exist_class(classlist):
    classcount = {}
    for classkind in classlist:
        if classkind not in classcount.keys():
            classcount[classkind] = 0
        classcount[classkind] += 1
    sortedclasscount = sorted(classcount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedclasscount[0][0]



def CreatTree(data, featurelist):
    FeaDistribution = [sample[-1] for sample in data]
    if len(set(FeaDistribution)) == 1:
        return FeaDistribution[0]
    if len(data[0]) == 1:
        return Most_exist_class(FeaDistribution)
    bestf = BestSplitingWay(data)
    best_f_in_list = featurelist[bestf]
    decisiontree = {best_f_in_list: {}}
    del (featurelist[bestf])
    kind = set([example[bestf] for example in data])
    for value in kind:
        subflist = featurelist[:]
        decisiontree[best_f_in_list][value] = CreatTree(Split_Data(data, bestf, value), subflist)
    return decisiontree



# decisionNode = dict(boxstyle = 'sawtooth',fc = '0.8')
# leafNode = dict(boxstyle = 'round4',fc = '0.8')
# arrow_args = dict(arrowstyle = '<-')

# def plotNode(nodeTxt,centerPt,parentPt,nodeType):
#    createPlot.ax1.annotate(nodeTxt,xy = parentPt,xycoords = 'axes fraction',xytext = centerPt,textcoords = 'axes fraction',va = 'center',
#                           ha = 'center',bbox=nodeType,arrowprops = arrow_args)

if __name__ == '__main__':
    file = open('data.txt')
    dataset = [inst.strip().split('\t') for inst in file.readlines()]
    label = ['age', 'prescrible', 'astigmatic', 'tearRate']
    print(CreatTree(dataset, label))

