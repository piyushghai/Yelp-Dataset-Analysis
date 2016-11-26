import pre_processing as pp

pp.resto_review['predicted_rating'] = round(sum(resto_review.stars_review)/len(resto_review.index))
print "Baseline Rating:", round(sum(resto_review.stars_review)/len(resto_review.index))

precision = metrics.precision_score(resto_review.stars_review, resto_review.predicted_rating)
recall = metrics.recall_score(resto_review.stars_review, resto_review.predicted_rating)
f1 = metrics.f1_score(resto_review.stars_review, resto_review.predicted_rating)
accuracy = accuracy_score(resto_review.stars_review, resto_review.predicted_rating)

baselineResult = {}

data = {'precision':precision,
        'recall':recall,
        'f1_score':f1,
        'accuracy':accuracy}

baselineResult['Baseline'] = data
pd.DataFrame(baselineResult).T
