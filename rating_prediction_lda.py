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

plt.rc('figure', figsize=(10,6))
seaborn.set()
colors = seaborn.color_palette()

#English Stop Words
stopwords = set(stopwords.words("english"))

totalTopics = 15

# Group all reviews per star rating and extract text out of them
starsGroup = resto_review.groupby('stars_review')

text_star_1 = starsGroup.get_group(1.0)['text']
text_star_2 = starsGroup.get_group(2.0)['text']
text_star_3 = starsGroup.get_group(3.0)['text']
text_star_4 = starsGroup.get_group(4.0)['text']
text_star_5 = starsGroup.get_group(5.0)['text']

# Optional : To reduce the dataset size and prevent laptop from frying, reduce the dataset size by sampling
#sampling = 10000 # No of rows to be sampled

#text_star_1 = text_star_1.sample(sampling)
#text_star_2 = text_star_2.sample(sampling)
#text_star_3 = text_star_3.sample(sampling)
#text_star_4 = text_star_4.sample(sampling)
#text_star_5 = text_star_5.sample(sampling)

label_star_1 = [1.0]*len(text_star_1)
label_star_2 = [2.0]*len(text_star_2)
label_star_3 = [3.0]*len(text_star_3)
label_star_4 = [4.0]*len(text_star_4)
label_star_5 = [5.0]*len(text_star_5)

# Create test and training dataset. We use 80-20 sampling here
from sklearn.cross_validation import train_test_split

train_stars_1, test_stars_1, train_labels_stars_1, all_1stars_labels_test = train_test_split(text_star_1, label_star_1, test_size=0.20)
train_stars_2, test_stars_2, train_labels_stars_2, all_2stars_labels_test = train_test_split(text_star_2, label_star_2, test_size=0.20)
train_stars_3, test_stars_3, train_labels_stars_3, all_3stars_labels_test = train_test_split(text_star_3, label_star_3, test_size=0.20)
train_stars_4, test_stars_4, train_labels_stars_4, all_4stars_labels_test = train_test_split(text_star_4, label_star_4, test_size=0.20)
train_stars_5, test_stars_5, train_labels_stars_5, all_5stars_labels_test = train_test_split(text_star_5, label_star_5, test_size=0.20)

#print(len(train_labels_stars_1))

# Clean all the reviews by removing stop words
def process_reviews(data_set):
    clean_data_set = []
    for text in data_set:
        # Remove punctuations
        text = re.sub(r'[^a-zA-Z]', ' ', review)
        # To lowercase
        text = review.lower()
        # Remove stop words
        texts = [word for word in text.lower().split() if word not in stopwords]
        try:
            clean_data_set.append(' '.join(texts))
        except:
            pass
    return clean_data_set

## Cleaning all the reviews and building corpus out of them
corpus_5stars = process_reviews(train_stars_5)
corpus_4stars = process_reviews(train_stars_4)
corpus_3stars = process_reviews(train_stars_3)
corpus_2stars = process_reviews(train_stars_2)
corpus_1stars = process_reviews(train_stars_1)

print "Number of 5-star reviews after processing: ", len(corpus_5stars)
print "Number of 4-star reviews after processing: ", len(corpus_4stars)
print "Number of 3-star reviews after processing: ", len(corpus_3stars)
print "Number of 2-star reviews after processing: ", len(corpus_2stars)
print "Number of 1-star reviews after processing: ", len(corpus_1stars)

# Creating combined dataset for training, containing representation of all the 5 star ratings possible
all_5_4_train = np.append(corpus_5stars, corpus_4stars)
all_5_4_3_train = np.append(all_5_4_train, corpus_3stars)
all_5_4_3_2_train = np.append(all_5_4_3_train, corpus_2stars)
all_text_train = np.append(all_5_4_3_2_train, corpus_1stars)


# Function to create the LDA model from the training dataset.
def perform_lda(train, totalTopics):
    corpus = []
    for review in train:
        # Remove punctuations
        review = re.sub(r'[^a-zA-Z]', ' ', review)
        # To lowercase
        review = review.lower()
        # Remove stop words
        texts = [word for word in review.lower().split() if word not in stopwords]
        try:
            corpus.append(texts)
        except:
            pass

    # Build dictionary
    dictionary = corpora.Dictionary(corpus)
    dictionary.save('restaurant_reviews.dict')
        
    # Build vectorized corpus
    corpus_2 = [dictionary.doc2bow(text) for text in corpus]
    #corpora.MmCorpus.serialize('LDA/restaurant_reviews.mm', corpus_2)
    
    lda = models.LdaModel(corpus_2, num_topics=totalTopics, id2word=dictionary)
    return lda

# Building the LDA model
%time lda = perform_lda(all_text_train, totalTopics)

# Generates a matrix of topic probabilities for each document in matrix
# Returns topic_dist for the input corpus, and all_dist, a running sum of all the corpuses
def generate_topic_dist_matrix(lda, totalTopics, corpus, all_dist, star):
    topic_dist = [0] * totalTopics
    dictionary = corpora.Dictionary.load("restaurant_reviews.dict")
    for doc in corpus:
        vec = dictionary.doc2bow(doc.lower().split())
        output = lda[vec]
        highest_prob = 0
        highest_topic = 0
        temp = [0] * totalTopics    # List to keep track of topic distribution for each document
        for topic in output:
            this_topic, this_prob = topic
            temp[this_topic] = this_prob
            if this_prob > highest_prob:
                highest_prob = this_prob 
                highest_topic = this_topic
        temp.append(star)
        all_dist.append(temp)
        topic_dist[highest_topic] += 1
    return topic_dist, all_dist

topic_dist_list = []

# Keep a separate list to count topics
topic_dist_5stars = []
topic_dist_4stars = []
topic_dist_3stars = []
topic_dist_2stars = []
topic_dist_1stars = []


topic_dist_5stars, topic_dist_list = generate_topic_dist_matrix(lda, totalTopics, corpus_5stars, topic_dist_list, 5)
topic_dist_4stars, topic_dist_list = generate_topic_dist_matrix(lda, totalTopics, corpus_4stars, topic_dist_list, 4)
topic_dist_3stars, topic_dist_list = generate_topic_dist_matrix(lda, totalTopics, corpus_3stars, topic_dist_list, 3)
topic_dist_2stars, topic_dist_list = generate_topic_dist_matrix(lda, totalTopics, corpus_2stars, topic_dist_list, 2)
topic_dist_1stars, topic_dist_list = generate_topic_dist_matrix(lda, totalTopics, corpus_1stars, topic_dist_list, 1)

cols = []
for i in xrange(1, totalTopics+1):
    cols.append("Topic"+ str(i))
cols.append("Star")

topic_dist_train_1_2_3_4_5_df = pd.DataFrame(topic_dist_list, columns=cols)

# Process the test reviews
corpus_5stars_test = process_reviews(test_stars_5)
corpus_4stars_test = process_reviews(test_stars_4)
corpus_3stars_test = process_reviews(test_stars_3)
corpus_2stars_test = process_reviews(test_stars_2)
corpus_1stars_test = process_reviews(test_stars_1)

print "Number of 5-star reviews after processing: ", len(corpus_5stars_test)
print "Number of 4-star reviews after processing: ", len(corpus_4stars_test)
print "Number of 3-star reviews after processing: ", len(corpus_3stars_test)
print "Number of 2-star reviews after processing: ", len(corpus_2stars_test)
print "Number of 1-star reviews after processing: ", len(corpus_1stars_test)



topic_dist_list = []

# Keep a separate list to count topics
topic_dist_5stars = []
topic_dist_4stars = []
topic_dist_3stars = []
topic_dist_2stars = []
topic_dist_1stars = []


topic_dist_5stars, topic_dist_list = generate_topic_dist_matrix(lda, totalTopics, corpus_5stars_test, topic_dist_list, 5)
topic_dist_4stars, topic_dist_list = generate_topic_dist_matrix(lda, totalTopics, corpus_4stars_test, topic_dist_list, 4)
topic_dist_3stars, topic_dist_list = generate_topic_dist_matrix(lda, totalTopics, corpus_3stars_test, topic_dist_list, 3)
topic_dist_2stars, topic_dist_list = generate_topic_dist_matrix(lda, totalTopics, corpus_2stars_test, topic_dist_list, 2)
topic_dist_1stars, topic_dist_list = generate_topic_dist_matrix(lda, totalTopics, corpus_1stars_test, topic_dist_list, 1)

cols = []
for i in xrange(1, totalTopics+1):
    cols.append("Topic"+ str(i))
cols.append("Star")

topic_dist_test_1_2_3_4_5_df = pd.DataFrame(topic_dist_list, columns=cols)

features = list(topic_dist_train_1_2_3_4_5_df.columns[:totalTopics])

x_train = topic_dist_train_1_2_3_4_5_df[features]
y_train = topic_dist_train_1_2_3_4_5_df['Star']

x_test = topic_dist_test_1_2_3_4_5_df[features]
y_test = topic_dist_test_1_2_3_4_5_df['Star'] 

clfs = [KNeighborsClassifier(), MultinomialNB(), LogisticRegression(), RandomForestClassifier(n_estimators=100, n_jobs=2), AdaBoostClassifier(n_estimators=100)]
clf_names = ['Nearest Neighbors', 'Multinomial Naive Bayes', 'Logistic Regression', 'Random Forest', 'AdaBoost']

LDAResults = {}
for (i, clf_) in enumerate(clfs):
    clf = clf_.fit(x_train, y_train)
    preds = clf.predict(x_test)
    
    precision = metrics.precision_score(y_test, preds)
    recall = metrics.recall_score(y_test, preds)
    f1 = metrics.f1_score(y_test, preds)
    accuracy = accuracy_score(y_test, preds)
    report = classification_report(y_test, preds)
    matrix = metrics.confusion_matrix(y_test, preds, labels=starsGroup.groups.keys())
    
    data = {'precision':precision,
            'recall':recall,
            'f1_score':f1,
            'accuracy':accuracy,
            'clf_report':report,
            'clf_matrix':matrix,
            'y_predicted':preds}
    
    LDAResults[clf_names[i]] = data

cols = ['precision', 'recall', 'f1_score', 'accuracy']
LDA_Prediction_Perf = pd.DataFrame(LDAResults).T[cols].T
LDA_Prediction_Perf

for model, val in LDAResults.iteritems():
    print '-------'+'-'*len(model)
    print 'MODEL:', model
    print '-------'+'-'*len(model)
    print 'The precision for this classifier is ' + str(val['precision'])
    print 'The recall for this classifier is    ' + str(val['recall'])
    print 'The f1 for this classifier is        ' + str(val['f1_score'])
    print 'The accuracy for this classifier is  ' + str(val['accuracy'])
    print 'The confusion matrix for this classifier is  \n' + str(val['clf_matrix'])
    print '\nHere is the classification report:'
    print val['clf_report']


