#! /usr/bin/python
# -*- coding:utf8 -*-

import numpy as np
import os, sys, getopt, time
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
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

    ## reproduce the baseline (Naive Bayes)
    # feature
    vectorizer = CountVectorizer(ngram_range=(1,2), binary=True)
    vectorizer = vectorizer.fit(x_train)
    x_train = vectorizer.transform(x_train).toarray()
    x_dev = vectorizer.transform(x_dev).toarray()
    x_test = vectorizer.transform(x_test).toarray()
    
    # build classifier
    clf = MultinomialNB()
    clf.fit(x_train, y_train)

    # classification & evalution
    predict = clf.predict(x_test)
    print ("error rate = ", np.mean(predict != y_test))
    
    ## parameter tuning
    step = 0.05
    nstep = int(1 / step)
    dev_err_cnt_alpha = []
    for alpha in range(1, nstep):
        clf = MultinomialNB(alpha=alpha*step)
        clf.fit(x_train, y_train)
        dev_err_cnt_alpha.append(np.mean(clf.predict(x_dev) != y_dev))
        print (dev_err_cnt_alpha[-1])

    best_alpha = step * (np.argmin(dev_err_cnt_alpha) + 1)
    print (best_alpha)

    x_train = np.concatenate((x_train, x_dev))
    y_train = np.concatenate((y_train, y_dev))
                
    clf = MultinomialNB(alpha=best_alpha)
    clf.fit(x_train, y_train)
    predict = clf.predict(x_test)
    print ("error rate = ", np.mean(predict != y_test))
        
        
    
