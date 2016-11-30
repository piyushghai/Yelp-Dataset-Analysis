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

# Function to create the LDA model from the training dataset.
def run_lda(corpus, totalTopics):

    # Build dictionary
    dictionary = corpora.Dictionary(corpus)

    # Save the dictionary
    dictionary.save('restaurant_reviews.dict')
        
    # Build vectorized corpus
    corpus_vector = [dictionary.doc2bow(text) for text in corpus]
    
    lda = models.LdaModel(corpus_vector, num_topics=totalTopics, id2word=dictionary)
    return lda

# Generates a list of topic probabilities for each document in the corpus
def getTopicProbDistMatrix(lda, num_topics, corpus, all_reviews_topic_dist):

    # Load the dictionary
    dictionary = corpora.Dictionary.load("restaurant_reviews.dict")

    # For every review in the corpus, compute the probability distribution matrix for each term
    for review in corpus:
  
        vec = dictionary.doc2bow(review.split())
        lda_output = lda[vec]

        review_topic_dist = [0] * num_topics    # List to store topic distribution for each review

        for topic in lda_output:
            topic_id, topic_prob = topic
            review_topic_dist[topic_id] = topic_prob

        all_reviews_topic_dist.append(review_topic_dist)

    return all_reviews_topic_dist


#Load the full training corpus
corpus_full_train = pickle.load (open("all_text_train.p", "rb"))

# Build tokenised corpus
corpus_tokenised = []
for text in corpus_full_train:
    text_list = [word for word in text]
    try:
        corpus_tokenised.append(text_list)
    except:
        pass

# Perform LDA on the tokenised training corpus
totalTopics = 7
lda = run_lda(corpus_tokenised, totalTopics);

#Load the star labels of the training and test dataset
stars_label_train_all_stars = pickle.load(open("all_stars_train.p","rb"))
stars_label_test_all_stars = pickle.load(open("all_stars_test.p","rb"))

#Load the training corpus for each star
corpus_5stars_train = pickle.load (open("corpus_5stars_train.p", "rb"))
corpus_4stars_train = pickle.load (open("corpus_4stars_train.p", "rb"))
corpus_3stars_train = pickle.load (open("corpus_3stars_train.p", "rb"))
corpus_2stars_train = pickle.load (open("corpus_2stars_train.p", "rb"))
corpus_1stars_train = pickle.load (open("corpus_1stars_train.p", "rb"))

#For all the training reviews, get the proabability distribution across the topics
topic_dist_list = []
topic_dist_list = getTopicProbDistMatrix(lda, totalTopics, corpus_5stars_train, topic_dist_list)
topic_dist_list = getTopicProbDistMatrix(lda, totalTopics, corpus_4stars_train, topic_dist_list)
topic_dist_list = getTopicProbDistMatrix(lda, totalTopics, corpus_3stars_train, topic_dist_list)
topic_dist_list = getTopicProbDistMatrix(lda, totalTopics, corpus_2stars_train, topic_dist_list)
topic_dist_list = getTopicProbDistMatrix(lda, totalTopics, corpus_1stars_train, topic_dist_list)

topic_dist_train_all_stars = pd.DataFrame(topic_dist_list)
pickle.dump(topic_dist_train_all_stars, open("topic_dist_train_all_stars.p", "wb"))

#Load the testing corpus for each star
corpus_5stars_test = pickle.load (open("corpus_5stars_test.p", "rb"))
corpus_4stars_test = pickle.load (open("corpus_4stars_test.p", "rb"))
corpus_3stars_test = pickle.load (open("corpus_3stars_test.p", "rb"))
corpus_2stars_test = pickle.load (open("corpus_2stars_test.p", "rb"))
corpus_1stars_test = pickle.load (open("corpus_1stars_test.p", "rb"))

#For all the testing reviews, get the proabability distribution across the topics
topic_dist_list = []
topic_dist_list = getTopicProbDistMatrix(lda, totalTopics, corpus_5stars_test, topic_dist_list)
topic_dist_list = getTopicProbDistMatrix(lda, totalTopics, corpus_4stars_test, topic_dist_list)
topic_dist_list = getTopicProbDistMatrix(lda, totalTopics, corpus_3stars_test, topic_dist_list)
topic_dist_list = getTopicProbDistMatrix(lda, totalTopics, corpus_2stars_test, topic_dist_list)
topic_dist_list = getTopicProbDistMatrix(lda, totalTopics, corpus_1stars_test, topic_dist_list)

topic_dist_test_all_stars = pd.DataFrame(topic_dist_list)
pickle.dump(topic_dist_test_all_stars, open("topic_dist_test_all_stars.p", "wb"))

# Train the classification algorithm models using topic distribution of the training reviews as features and star rating as the label.
# Use the model to predict star rating from the topic distribution of the testing reviews
train_features = topic_dist_train_all_stars
train_labels = stars_label_train_all_stars

test_features = topic_dist_test_all_stars
test_lables = stars_label_test_all_stars

classifiers = [MultinomialNB(), LogisticRegression(), RandomForestClassifier(n_estimators=100, n_jobs=2), AdaBoostClassifier(n_estimators=100)]
classifiers_names = ['Multinomial Naive Bayes', 'Logistic Regression', 'Random Forest', 'AdaBoost']

LDA_Pred_Results = {}
for (i, classifier) in enumerate(classifiers):
    model = classifier.fit(train_features, train_labels)
    prediction = model.predict(test_features)
    
    precision = metrics.precision_score(test_lables, prediction)
    recall = metrics.recall_score(test_lables, prediction)
    F1 = metrics.f1_score(test_lables, prediction)
    accuracy = accuracy_score(test_lables, prediction)
    report = classification_report(test_lables, prediction)
    matrix = metrics.confusion_matrix(test_lables, prediction, labels=[1.0, 2.0, 3.0, 4.0, 5.0])
    
    data = {'precision':precision,
            'recall':recall,
            'F1_score':F1,
            'accuracy':accuracy,
            'classifierreport':report,
            'classifiermatrix':matrix,
            'y_predicted':prediction}
    
    LDA_Pred_Results[classifiers_names[i]] = data

sys.stdout = open("lda_results.txt", 'a')

for model, val in LDA_Pred_Results.iteritems():
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

