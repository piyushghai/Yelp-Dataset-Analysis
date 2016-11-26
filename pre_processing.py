import pandas as pd
import re

pyplot.rc('figure', figsize=(10,6))
seaborn.set()
colors = seaborn.color_palette()

## Reading in the dataset. The dataset's already been preprocessed and converted to csv from json structure
business_data = pd.read_csv("Data/yelp_academic_dataset_business.csv", dtype=unicode)
review_data = pd.read_csv("Data/yelp_academic_dataset_review.csv")

#len(review_data)
#len(business_data)
#Filtering out on restaurants from the dataset. 
business_data = business_data[business_data['categories'].str.contains('Restaurants')]
# Joining the reviews for a restaurant with the restaurant categories obtained above. 
# This essentially means, given a business id, what are all the reviews for it.

review_data = review_data.merge(business_data,
                                         left_on='business_id',
                                         right_on='business_id',
                                         suffixes=('_review', '_business'))

# Further filtering. We are only interested in text of reviews and the awarded rating in a review
review_data = review_data.ix[:,['text','stars_review']]

# Dropping rows corresponding to NA column vals.
review_data = review_data.text.dropna()

# Display the histogram for length of a review v/s count
review_data.text.str.len().hist(bins=50)
xlabel('Length of the review')
ylabel('Number of reviews')

minReviewLen = 0
resto_review_reduced = review_data[review_data.text.str.len() > minReviewLen]

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