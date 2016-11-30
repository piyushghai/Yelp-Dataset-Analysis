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

#Load the full training and testing corpus
corpus_full_train = pickle.load (open("all_text_train.p", "rb"))
corpus_full_test = pickle.load (open("all_text_test.p", "rb"))

#Load the star labels of the training and test review
stars_label_train_all_stars = pickle.load(open("all_stars_train.p","rb"))
stars_label_test_all_stars = pickle.load(open("all_stars_test.p","rb"))

#Load the  topic distribution of the training and test reviews
topic_dist_train_all_stars = pickle.load(open("topic_dist_train_all_stars.p","rb"))
topic_dist_test_all_stars = pickle.load(open("topic_dist_test_all_stars.p","rb"))

# Defining Sentiment : If review is < 3 stars, label it as -ve sentiment, else +ve sentiment
def getSentiment(x):
    if x < 3.0:
        return 0
    else:
        return 1

# Use Logistic Regression classifier for predicting the sentiment of review
# Train the classifier on the TFIDF representation of the corpus

tfidfvectorizer = TfidfVectorizer()

topic_dist_train_all_stars['Sentiment'] = topic_dist_train_all_stars.map(getSentiment)
topic_dist_test_all_stars['Sentiment'] = topic_dist_test_all_stars.map(getSentiment)

tfidf_train = tfidfvectorizer.fit_transform(corpus_full_train)
tfidf_test = tfidfvectorizer.transform(corpus_full_test)

sentimentLabel_train = topic_dist_train_all_stars['Sentiment']

classifier = LogisticRegression().fit(tfidf_train, sentimentLabel_train)

sentiment_predicted_train = classifier.predict(tfidf_train)
sentiment_predicted_test = classifier.predict(tfidf_test)

topic_dist_train_all_stars['Sentiment'] = sentiment_predicted_train
topic_dist_test_all_stars['Sentiment'] = sentiment_predicted_test

# Feed in the predicted sentiment as the feature along with the topic distribution (From LDA), for the model to train on
# Use the model to predict star rating from the topic distribution and sentiment of the testing reviews 
train_features = topic_dist_train_all_stars
train_lables = stars_label_train_all_stars

test_features = topic_dist_test_all_stars
test_lables = stars_label_test_all_stars 

classifiers = [MultinomialNB(), LogisticRegression(), RandomForestClassifier(n_estimators=100, n_jobs=2), AdaBoostClassifier(n_estimators=100)]
classifiers_names = ['Multinomial Naive Bayes', 'Logistic Regression', 'Random Forest', 'AdaBoost']

LdaSentimentResults = {}
for (i, classifier) in enumerate(classifiers):
    model = classifier.fit(train_features, train_lables)
    preds = model.predict(test_features)
    
    precision = metrics.precision_score(test_lables, preds)
    recall = metrics.recall_score(test_lables, preds)
    F1 = metrics.f1_score(test_lables, preds)
    accuracy = accuracy_score(test_lables, preds)
    report = classification_report(test_lables, preds)
    matrix = metrics.confusion_matrix(test_lables, preds, labels=[1.0, 2.0, 3.0, 4.0, 5.0])
    
    data = {'precision':precision,
            'recall':recall,
            'f1_score':f1,
            'accuracy':accuracy,
            'classifierreport':report,
            'classifiermatrix':matrix,
            'y_predicted':prediction}
    
    LdaSentimentResults[classifiers_names[i]] = data

sys.stdout = open("lda_sentiment_results.txt", 'a')

for model, val in LdaSentimentResults.iteritems():
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