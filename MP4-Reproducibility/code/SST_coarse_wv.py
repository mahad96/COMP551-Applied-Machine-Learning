#! /usr/bin/python
# -*- coding:utf8 -*-

import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction.text import CountVectorizer

def initWfromFile(filename, d, vocab):
    '''
    loading word vectors
    '''
    initW = {}
    f = open(filename, 'r')	
    for line in f.readlines():
        word = line[:line.index(' ')]
        if word in vocab:
            initW[word] = np.array(list(map(float, line[line.index(' ')+1:].split(' '))))

    return initW

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

def ave_vector(x, W):
    d = 100
    v = np.zeros((len(x), d))
    for i, _x in enumerate(x):
        num = 0
        for word in _x.split():
            if word in W:
                num += 1.
                v[i,:] += W[word]
        if num > 0:
            v[i,:] /= num

    return v

if __name__ == '__main__':
    x_train, y_train = read_data('sst-data/sst5_train_sentences.csv')
    x_dev, y_dev = read_data('sst-data/sst5_dev.csv')
    x_test , y_test  = read_data('sst-data/sst5_test.csv')
    y_train = cls2ind(y_train)
    y_test  = cls2ind(y_test)
    y_dev   = cls2ind(y_dev)

    # feature
    vectorizer = CountVectorizer(ngram_range=(1,1), binary=True)
    vectorizer = vectorizer.fit(list(x_train)+list(x_dev)+list(x_test))
    vocab = vectorizer.vocabulary_
    W = initWfromFile('./glove.6B.100d.txt', 100, vocab)
    
    x_train = ave_vector(x_train, W)
    x_dev = ave_vector(x_dev, W)
    x_test = ave_vector(x_test, W)

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
    clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=5000)
    clf.fit(x_train, y_train)

    # classification & evalution
    predict = clf.predict(x_test)
    print ("error rate = ", np.mean(predict != y_test))

    ## parameter tuning
    step = 0.0001
    nstep = int(0.001 / step)
    dev_err_cnt_alpha = []
    for alpha in range(1, nstep):
        clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=5000, alpha=alpha*step)
        clf.fit(x_train, y_train)
        dev_err_cnt_alpha.append(np.mean(clf.predict(x_dev) != y_dev))

    best_alpha = step * (np.argmin(dev_err_cnt_alpha) + 1)
    print (best_alpha)
                
    clf = SGDClassifier(loss="hinge", penalty="l2", max_iter=5000, alpha=best_alpha)
    clf.fit(x_train, y_train)
    predict = clf.predict(x_test)
    print ("error rate = ", np.mean(predict != y_test))
        
        
# output
#error rate =  0.240527182867
#0.0001
#error rate =  0.238330587589


