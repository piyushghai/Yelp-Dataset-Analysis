import time
import datetime

import cPickle as pickle
import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
import pylab
import re
import scipy as sp
import seaborn

from gensim import corpora, models
from nltk.corpus import stopwords
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.lda import LDA
from sklearn.qda import QDA
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import *
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix, classification_report, precision_score, recall_score, roc_auc_score
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from collections import Counter
import os
import sys

# Load the corpus and star labels for training and testing dataset
corpus_full_train = pickle.load (open("all_text_train.p", "rb"))
corpus_full_test = pickle.load (open("all_text_test.p", "rb"))
stars_label_train = pickle.load(open("all_stars_train.p","rb"))
stars_label_test = pickle.load(open("all_stars_test.p","rb"))

# Extracting features using term frequency
tfidfvectorizer = TfidfVectorizer()

tfidf_train = tfidfvectorizer.fit_transform(corpus_full_train)
tfidf_test = tfidfvectorizer.transform(corpus_full_test)

classifiers = [MultinomialNB(), LogisticRegression()]
classifiers_names = ['Multinomial Naive Bayes', 'Logistic Regression']

TFIDF_Pred_Results = {}

# Train the two classifiers on the TFIDF representation of the corpus and obatin the metrics on the test dataset
for (i, classifier) in enumerate(classifiers):
    model = classifier.fit(tfidf_train, stars_label_train)
    prediction = model.predict(tfidf_test)
    
    precision = metrics.precision_score(stars_label_test, prediction)
    recall = metrics.recall_score(stars_label_test, prediction)
    F1 = metrics.f1_score(stars_label_test, prediction)
    accuracy = accuracy_score(stars_label_test, prediction)
    report = classification_report(stars_label_test, prediction)
    matrix = metrics.confusion_matrix(stars_label_test, prediction, labels=[1.0, 2.0, 3.0, 4.0, 5.0])
    
    data = {'precision':precision,
            'recall':recall,
            'F1_score':F1,
            'accuracy':accuracy,
            'classifierreport':report,
            'classifiermatrix':matrix,
            'y_predicted':prediction}
    
    TFIDF_Pred_Results[classifiers_names[i]] = data

cols = ['precision', 'recall', 'f1_score', 'accuracy']
pd.DataFrame(TFIDF_Pred_Results).T[cols].T

sys.stdout = open("tfidf_results.txt", 'a')

for model, val in TFIDF_Pred_Results.iteritems():
    print '-------'+'-'*len(model)
    print 'MODEL:', model
    print '-------'+'-'*len(model)
    print ('Precision = ' + str(precision))
    print ('Recall = ' + str(recall))
    print ('F1 = ' + str(F1))
    print ('Accuracy = ' + str(accuracy))
    print 'Confusion matrix =  \n' + str(val['classifiermatrix'])
    print '\nHere is the classification report:'
    print val['classifierreport']