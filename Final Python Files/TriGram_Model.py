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
from sklearn.cross_validation import train_test_split
import os

## Load the previously saved restaurant review data
review_data = pickle.load( open( "resto_review.p", "rb" ) )

### Optional : Sampling : For testing using smaller dataset
#review_data = review_data.sample(5000)

train_X, test_X, train_y, test_y = train_test_split(review_data.text, review_data.stars_review, test_size=0.30)

trigram_vectorizer = CountVectorizer(analyzer = "word",
                                     tokenizer = None,
                                     preprocessor = None,
                                     ngram_range = (3, 3),
                                     strip_accents='unicode')

trigram_feature_matrix_train = trigram_vectorizer.fit_transform(train_X)
trigram_feature_matrix_test = trigram_vectorizer.transform(test_X)

tri_gram_multinomial_NB_classifier = MultinomialNB().fit(trigram_feature_matrix_train, train_y)
tri_gram_multinomial_NB_prediction = tri_gram_multinomial_NB_classifier.predict(trigram_feature_matrix_test)

model = 'Trigram Multinomial Naive Bayes'
target_names = ['1 star', '2 star', '3 star', '4 star', '5 star']

sys.stdout = open("trigram_results.txt", 'a')

print '-------'+'-'*len(model)
print 'MODEL:', model
print '-------'+'-'*len(model)

print 'Precision = ' + str(metrics.precision_score(test_y, tri_gram_multinomial_NB_prediction))
print 'Recall = ' + str(metrics.recall_score(test_y, tri_gram_multinomial_NB_prediction))
print 'F1 = ' + str(metrics.f1_score(test_y, tri_gram_multinomial_NB_prediction))
print 'Accuracy = ' + str(metrics.accuracy_score(test_y, tri_gram_multinomial_NB_prediction))
print 'Confusion matrix =  \n' + str(metrics.confusion_matrix(test_y, tri_gram_multinomial_NB_prediction, labels=[1.0, 2.0, 3.0, 4.0, 5.0]))
print '\nClassification Report:\n' + classification_report(test_y, tri_gram_multinomial_NB_prediction, target_names=target_names)

