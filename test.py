import sys
import svm as svm
import loader
import libsvm2csv
#from sklearn.cross_validation import KFold
import numpy as np


tr_input_file = r'traindata.csv'
#tr_inputlabel_file = r'trainlabel.csv'
tst_input_file = r'testdata.csv'

d = 57

# # used for tuning with k-fold only 
# # labels = np.asarray(loader.label_loader(tr_file).keys())
# # kf=KFold(n=len(labels),n_folds=5,shuffle=True)
# # for tr,tst in kf:
# #     trset = labels[tr] #get row
# #     tstset = labels[tst] #get row

# #read file and transform into csv
# libsvm2csv.getcsv(tr_input_file, tr_output_file,d)
# libsvm2csv.getcsv(tst_input_file, tst_output_file,d)

#process svm data
#labels = np.asarray(loader.label_loader(tr_inputlabel_file).keys())
# print labels[200]
svm_trainset = loader.train_feature_loader(tr_input_file)
svm_testset = loader.test_feature_loader(tst_input_file)
svm_tr_set_feature = svm_trainset['featureList']
svm_tr_set_label = svm_trainset['labelList']
svm_tst_set_feature = [svm_testset['featureList'][row] for row in labels]
svm_tst_set_label = [svm_testset['labelList'][row] for row in labels]

svm.train(svm_tr_set_feature,svm_tr_set_label)
svm.classify(svm_tst_set_feature,svm_tst_set_label)




