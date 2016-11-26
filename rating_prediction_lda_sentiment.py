from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import accuracy_score

from rating_prediction_lda import totalTopics,all_text_train, all_text_test,topic_dist_train_all_stars,topic_dist_test_all_stars
from rating_prediction_tfidf import tfidfvectorizer

def getSentiment(s):
    if s < 3.5:
        return 0
    else:
        return 1


topic_dist_train_all_stars['Sentiment'] = topic_dist_train_all_stars['Star'].map(getSentiment)
topic_dist_test_all_stars['Sentiment'] = topic_dist_test_all_stars['Star'].map(getSentiment)

sentimentTextTrain = tfidfvectorizer.fit_transform(all_text_train)
sentimentTextTest = tfidfvectorizer.transform(all_text_test)

sentimentLabelTrain = topic_dist_train_all_stars['Sentiment']
sentimentLabelTest = topic_dist_test_all_stars['Sentiment']

classifier = LogisticRegression().fit(sentimentTextTrain, sentimentLabelTrain)

ySentimentTrain = classifier.predict(sentimentTextTrain)
ySentimentTest = classifier.predict(sentimentTextTest)

topic_dist_train_all_stars['Sentiment_Predicted'] = ySentimentTrain
topic_dist_test_all_stars['Sentiment_Predicted'] = ySentimentTest


features = list(topic_dist_train_all_stars.columns[:totalTopics])
features.append(topic_dist_train_all_stars.columns[totalTopics+2])


x_train = topic_dist_train_all_stars[features]
y_train = topic_dist_train_all_stars['Star']

x_test = topic_dist_test_all_stars[features]
y_test = topic_dist_test_all_stars['Star'] 

classifiers = [MultinomialNB(), LogisticRegression(), RandomForestClassifier(n_estimators=100, n_jobs=2), AdaBoostClassifier(n_estimators=100)]
classifiers_names = ['Multinomial Naive Bayes', 'Logistic Regression', 'Random Forest', 'AdaBoost']

LdaSentimentResults = {}
for (i, clf_) in enumerate(classifiers):
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
    
    LdaSentimentResults[classifiers_names[i]] = data



cols = ['precision', 'recall', 'f1_score', 'accuracy']
pd.DataFrame(LdaSentimentResults).T[cols].T

for model, val in LdaSentimentResults.iteritems():
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
