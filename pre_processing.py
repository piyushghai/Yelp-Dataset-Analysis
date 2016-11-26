import time
import datetime

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

## Reading in the dataset. The dataset's already been preprocessed and converted to csv from json structure
business_data = pd.read_csv("Data/yelp_academic_dataset_business.csv", dtype=unicode)
review_data = pd.read_csv("Data/yelp_academic_dataset_review.csv")

#len(review_data)
#len(business_data)
#Filtering out on restaurants from the dataset. 
resto_business_data = business_data[business_data['categories'].str.contains('Restaurants')]
# Joining the reviews for a restaurant with the restaurant categories obtained above. 
# This essentially means, given a business id, what are all the reviews for it.

resto_review_data = review_data.merge(resto_business_data,
                                         left_on='business_id',
                                         right_on='business_id',
                                         suffixes=('_review', '_business'))

# Further filtering. We are only interested in text of reviews and the awarded rating in a review
resto_review_data = resto_review_data.ix[:,['text','stars_review']]

# Dropping rows corresponding to NA column vals.
resto_review_data = resto_review_data.text.dropna()

# Display the histogram for length of a review v/s count
resto_review_data.text.str.len().hist(bins=50)
xlabel('Length of the review')
ylabel('Number of reviews')


minReviewLen = 0
resto_review_reduced = resto_review_data[resto_review_data.text.str.len() > minReviewLen]

maxReviewLen = 500  #Gives approximately a million records. We can do this for entire range too, but that would be ~2 million records and fried laptop!
resto_review_reduced = resto_review_reduced[resto_review_reduced.text.str.len() < maxReviewLen]

# Converting it to float for further model processing
resto_review.stars_review = resto_review.stars_review.astype(float)

#Histogram for number of reviews v/s their rating
bins = [1, 2, 3 ,4, 5, 6]
resto_review.stars_review.hist(bins=bins, align='left', width=0.93)
xticks(bins)
xlabel('Rating stars')
ylabel('Number of reviews')


