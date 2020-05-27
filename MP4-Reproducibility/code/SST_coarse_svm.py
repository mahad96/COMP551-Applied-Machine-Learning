#! /usr/bin/python
# -*- coding:utf8 -*-

import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.model_selection import StratifiedKFold

def read_data(csv_file):
    lines = open(csv_file).readlines()
    x, y = [], []
    for line in lines:
        line = line.strip().rsplit(',', 1)
        x.append(line[0])
        y.append(line[1])
        
    return x, y

def cls2ind(y):
    cls = {'very neg': -2, 
           'neg': -1,
           'neu': 0,
           'pos': 1,
           'very pos': 2}
    y_ind = [cls[c] for c in y]
    return y_ind

if __name__ == '__main__':
    x_train, y_train = read_data('sst-data/sst5_train_sentences.csv')
    x_dev, y_dev = read_data('sst-data/sst5_dev.csv')
    x_test , y_test  = read_data('sst-data/sst5_test.csv')
    y_train = cls2ind(y_train)
    y_test  = cls2ind(y_test)
    y_dev   = cls2ind(y_dev)
 
    # feature
    vectorizer = CountVectorizer(ngram_range=(1,2), binary=True, max_features=10000)
    vectorizer = vectorizer.fit(x_train + x_dev + x_test)
    x_train = vectorizer.transform(x_train).toarray()
    x_dev = vectorizer.transform(x_dev).toarray()
    x_test = vectorizer.transform(x_test).toarray()
    
    # coarse
    y_train, y_dev, y_test = np.array(y_train), np.array(y_dev), np.array(y_test)
    y_train[y_train < 0] = -1
    y_dev[y_dev < 0] = -1
    y_test[y_test < 0] = -1
    y_train[y_train > 0] = 1
    y_dev[y_dev > 0] = 1
    y_test[y_test > 0] = 1
    x_train = x_train[y_train != 0]
    x_dev = x_dev[y_dev != 0]
    x_test = x_test[y_test != 0]
    y_train = y_train[y_train != 0]
    y_dev = y_dev[y_dev != 0]
    y_test = y_test[y_test != 0]

    # build classifier
    clf = LinearSVC()
    clf.fit(x_train, y_train)
    
    # classification & evalution
    predict = clf.predict(x_test)
    print ("error rate = ", np.mean(predict != y_test))
    
    ## parameter tuning
    Clist = [0.01, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    dev_err_cnt_C = []
    for _C in Clist:
        clf = LinearSVC(C=_C)
        clf.fit(x_train, y_train)
        dev_err_cnt_C.append(np.mean(clf.predict(x_dev) != y_dev))

    best_C = Clist[np.argmin(dev_err_cnt_C)]
    print (best_C)
    
    x_train = np.concatenate((x_train, x_dev))
    y_train = np.concatenate((y_train, y_dev))
                
    clf = LinearSVC(C=best_C)
    clf.fit(x_train, y_train)
    predict = clf.predict(x_test)
    print ("error rate = ", np.mean(predict != y_test))

#     outputs:
# error rate =  0.226249313564
# 0.1
# error rate =  0.203734211971
# SVMs (Socher et al., 2013b) 20.6%