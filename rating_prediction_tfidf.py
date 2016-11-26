from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from rating_prediction_lda import starsGroup,all_text_train, all_text_test,topic_dist_train_all_stars,topic_dist_test_all_stars

# Extracting features using term frequency
tfidfvectorizer = TfidfVectorizer()

tfidfTrain = tfidfvectorizer.fit_transform(all_text_train)
tfidfTest = tfidfvectorizer.transform(all_text_test)

tfidfLabelTrain = topic_dist_train_all_stars['Star']
tfidfLabelTest = topic_dist_train_all_stars['Star']

classifiers = [MultinomialNB(), LogisticRegression()]
classifiers_names = ['Multinomial Naive Bayes', 'Logistic Regression']

TFIDF_Pred_Results = {}
for (i, clf_) in enumerate(classifiers):
    clf = clf_.fit(tfidfTrain, tfidfLabelTrain)
    prediction = clf.predict(tfidfTest)
    
    precision = metrics.precision_score(tfidfLabelTrain, prediction)
    recall = metrics.recall_score(tfidfLabelTest, prediction)
    f1 = metrics.f1_score(tfidfLabelTest, prediction)
    accuracy = accuracy_score(tfidfLabelTest, prediction)
    report = classification_report(tfidfLabelTest, prediction)
    matrix = metrics.confusion_matrix(tfidfLabelTest, prediction, labels=starsGroup.groups.keys())
    
    data = {'precision':precision,
            'recall':recall,
            'f1_score':f1,
            'accuracy':accuracy,
            'clf_report':report,
            'clf_matrix':matrix,
            'y_predicted':preds}
    
    TFIDF_Pred_Results[classifiers_names[i]] = data

cols = ['precision', 'recall', 'f1_score', 'accuracy']
pd.DataFrame(TFIDF_Pred_Results).T[cols].T



for model, val in TFIDF_Pred_Results.iteritems():
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

