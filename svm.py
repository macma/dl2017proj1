from sklearn.decomposition import PCA
from sklearn.svm import SVC
import numpy as np
from sklearn.externals import joblib

def train(tr_set_feature,tr_set_label):
    pca_model_path = r'trained\pca.pkl'
    normalization_path=r'trained\norm.pkl'
    svm_model_path = r'trained\svm.pkl'

    #PCA
    pca = PCA(n_components=50)
    tr_set_feature = pca.fit_transform(tr_set_feature)
    print "PCA explains variance ratio: "
    print sum(pca.explained_variance_ratio_)
    #print pca.explained_variance_ratio_

    joblib.dump(pca, pca_model_path)

    #Normalization
    feat_mean = np.mean(tr_set_feature,axis=0)
    feat_std = np.std(tr_set_feature,axis=0)
    tr_set_feature -= feat_mean
    tr_set_feature /= feat_std
    #save normalization
    content = {"mean":feat_mean,"std":feat_std}
    joblib.dump(content,normalization_path)

    #Training SVM Model
    model = SVC(C=5, probability=True)
    model.fit(tr_set_feature, tr_set_label)

    #Measuring training and test accuracy
    tr_accuracy = np.mean(np.argmax(model.predict_proba(tr_set_feature),axis=1) == tr_set_label)

    print "SVM Train Accuracy:%f" % (tr_accuracy)

    joblib.dump(model, svm_model_path)


def classify(tst_set_feature):#,tst_set_label):
    pca_model_path = r'trained\pca.pkl'
    normalization_path=r'trained\norm.pkl'
    svm_model_path = r'trained\svm.pkl'

    pca = joblib.load(pca_model_path)

    normalization = joblib.load(normalization_path)
    feat_mean = normalization['mean']
    feat_std = normalization['std']

    svm = joblib.load(svm_model_path)
    print (type(tst_set_feature))
    features = pca.transform(tst_set_feature)

    features -= feat_mean
    features /= feat_std
    
    tst_predicted = svm.predict(features)
    print(tst_predicted)
    l = []
    for a in tst_predicted:
        l.append(str(a) + ".0\n")
    s = "".join(l)
    f = open('result.txt','w')
    f.write(s)
    f.close()
    #tst_accuracy = np.mean(np.argmax(tst_predicted,axis=1)== tst_set_label)

    #print "SVM Classification Accuracy:%f" % (tst_accuracy)


