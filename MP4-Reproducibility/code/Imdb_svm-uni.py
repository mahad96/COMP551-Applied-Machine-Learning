#! /usr/bin/python
# -*- coding:utf8 -*-

import numpy as np
import os
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer

def read_data(path):
    x, y = [], []
    for file in os.listdir(os.path.join(path, 'pos')):
        line = open(os.path.join(path, 'pos', file)).readline()
        line = line.strip()
        x.append(line)
        y.append(1)
    for file in os.listdir(os.path.join(path, 'neg')):
        line = open(os.path.join(path, 'neg', file)).readline()
        line = line.strip()
        x.append(line)
        y.append(0)
    
    return x, y

if __name__ == '__main__':
    x_train, y_train = read_data('./aclImdb/train/')
    x_test , y_test  = read_data('./aclImdb/test/')
    
    # feature
    vectorizer = CountVectorizer(ngram_range=(1,1), binary=True, max_features=5000)
    vectorizer = vectorizer.fit(x_train + x_test)
    x_train = vectorizer.transform(x_train).toarray()
    x_test = vectorizer.transform(x_test).toarray()
    y_train, y_test = np.array(y_train), np.array(y_test)
    
    # build classifier
    clf = LinearSVC(C=.01)
    clf.fit(x_train, y_train)

    # classification & evalution
    predict = clf.predict(x_test)
    print ("error rate = ", np.mean(predict != y_test))
    
    # output: error rate =  0.12388 (SVM-uni (Wang & Manning, 2012) 13.05%)