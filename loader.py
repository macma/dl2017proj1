
import os
import numpy as np

'''
return a dictionary of [{row:,label:}]

'''
def label_loader(input_file):
    allSeries = {}
    row = 1
    with open(input_file,'r') as f:
        for line in f: #read in record
            # tuples = line.split(",")
            key = row
            val = 1 if(line[0:3]=='1.0') else 0
            # print (line[0:3])
            allSeries[key]= val
            # if(key == 10):
            #     print allSeries[10]
            row+=1
    print allSeries[100]
    return allSeries

'''
input training file
output  training feature
        training label
'''
def train_feature_loader(input_file):
    tr_featureList = []
    tr_labelList = []
    with open(input_file,'r') as f:
        for line in f: #read in record
            tuples = line.split(",")
            featurestr = tuples[1:len(tuples)]
            features = map(lambda x:float(x),featurestr)
            label = 1 if(tuples[0]=='1.0') else 0

            tr_featureList.append(features)
            tr_labelList.append(label)

    return {"featureList":tr_featureList,"labelList":tr_labelList}

'''
input testing file
output  testing feature
        testing label
'''
def test_feature_loader(input_file):
    tst_featureList = {}
    tst_labelList = {}
    row = 1
    with open(input_file,'r') as f:
        for line in f: #read in record
            tuples = line.split(",")
            featurestr = tuples[0:len(tuples)]
            features = map(lambda x:float(x),featurestr)
            label = 1# if(tuples[0]=='1.0') else 0


            tst_featureList[row] =  features
            tst_labelList[row] = label
            row+=1
    return {"featureList":tst_featureList,"labelList":tst_labelList}