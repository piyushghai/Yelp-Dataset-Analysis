import cPickle as pickle
from sklearn import metrics
from sklearn.metrics import accuracy_score
import pandas as pd
import sys


resto_review_data = pickle.load(open("resto_review.p", "rb"))

# Assigning predicted rating as the average rating of all reviews
resto_review_data['predicted_rating'] = round(sum(resto_review_data.stars_review)/len(resto_review_data.index))

precision = metrics.precision_score(resto_review_data.stars_review, resto_review_data.predicted_rating)
recall = metrics.recall_score(resto_review_data.stars_review, resto_review_data.predicted_rating)
F1 = metrics.f1_score(resto_review_data.stars_review, resto_review_data.predicted_rating)
accuracy = accuracy_score(resto_review_data.stars_review, resto_review_data.predicted_rating)

baselineResult = {}

data = {'precision':precision,
        'recall':recall,
        'f1_score':f1,
        'accuracy':accuracy}

baselineResult['Baseline'] = data

sys.stdout = open("baseline_results.txt", 'a')

model = "Baseline"
print ('-------'+'-'*len(model))
print ('MODEL:' + model)
print ('-------'+'-'*len(model))
print ('Precision = ' + str(precision))
print ('Recall = ' + str(recall))
print ('F1 = ' + str(F1))
print ('Accuracy = ' + str(accuracy))
