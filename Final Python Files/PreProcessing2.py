from nltk.corpus import stopwords
import cPickle as pickle
import string
import nltk
from nltk.stem.porter import PorterStemmer
import pandas as pd
from sklearn.cross_validation import train_test_split

NLTK_STOPWORDS = set(stopwords.words('english'))
MORE_STOPWORDS = set([line.strip() for line in open('more_stopwords.txt', 'r')])

# Functions needed for the NLP Pre-Processing
def lowercase(s):
    return s.lower()

def tokenize(s):
    token_list = nltk.word_tokenize(s)
    return token_list

def remove_punctuation(s):
    return s.translate(None, string.punctuation)

def remove_numbers(s):
    return s.translate(None, string.digits)
 
def remove_stopwords(token_list):
    exclude_stopwords = lambda token : token not in NLTK_STOPWORDS
    return filter(lambda tok : tok not in MORE_STOPWORDS, filter(exclude_stopwords, token_list))

def stemming_token_list(token_list):
    STEMMER = PorterStemmer()
    return [STEMMER.stem(tok.decode('utf-8')) for tok in token_list]

def restring_tokens(token_list):
    return ' '.join(token_list)

# Function to clean the reviews using the Pre-Processing functions written above
def clean_reviews(data_set):
    clean_data_set = []
    for text in data_set:
        text = lowercase(text)
        text = remove_punctuation(text)
        text = remove_numbers(text)

        token_list = tokenize(text)
        token_list = remove_stopwords(token_list)

        token_list = stemming_token_list(token_list)
        
        try:
            clean_data_set.append(restring_tokens(token_list))
        except:
            pass
    return clean_data_set

## Load the previously saved pickle file
resto_review_data = pickle.load( open( "resto_review.p", "rb" ) )

### Optional :  Sampling : For testing using smaller dataset
#resto_review_data = resto_review_data.sample(5000)

# Group all reviews per star rating and extract text out of them
resto_review_starGrouping = resto_review_data.groupby('stars_review')

review_text_star_1 = resto_review_starGrouping.get_group(1.0)['text']
review_text_star_2 = resto_review_starGrouping.get_group(2.0)['text']
review_text_star_3 = resto_review_starGrouping.get_group(3.0)['text']
review_text_star_4 = resto_review_starGrouping.get_group(4.0)['text']
review_text_star_5 = resto_review_starGrouping.get_group(5.0)['text']

### Optional : Sampling : For testing using smaller dataset
#sampling = 5000 # No of rows to be sampled
#review_text_star_1 = review_text_star_1.sample(sampling)
#review_text_star_2 = review_text_star_2.sample(sampling)
#review_text_star_3 = review_text_star_3.sample(sampling)
#review_text_star_4 = review_text_star_4.sample(sampling)
#review_text_star_5 = review_text_star_5.sample(sampling)

# Add all the corresponding original labels to reviews
review_label_star_1 = [1.0]*len(review_text_star_1)
review_label_star_2 = [2.0]*len(review_text_star_2)
review_label_star_3 = [3.0]*len(review_text_star_3)
review_label_star_4 = [4.0]*len(review_text_star_4)
review_label_star_5 = [5.0]*len(review_text_star_5)

# We create a 70-30 partition to create training and test dataset
train_review_text_star_1, test_review_text_star_1, train_labels_stars_1, test_labels_stars_1 = train_test_split(review_text_star_1, review_label_star_1, test_size=0.30)
train_review_text_star_2, test_review_text_star_2, train_labels_stars_2, test_labels_stars_2 = train_test_split(review_text_star_2, review_label_star_2, test_size=0.30)
train_review_text_star_3, test_review_text_star_3, train_labels_stars_3, test_labels_stars_3 = train_test_split(review_text_star_3, review_label_star_3, test_size=0.30)
train_review_text_star_4, test_review_text_star_4, train_labels_stars_4, test_labels_stars_4 = train_test_split(review_text_star_4, review_label_star_4, test_size=0.30)
train_review_text_star_5, test_review_text_star_5, train_labels_stars_5, test_labels_stars_5 = train_test_split(review_text_star_5, review_label_star_5, test_size=0.30)

## Cleaning all the training reviews and building corpus out of them
corpus_5stars_train = clean_reviews(train_review_text_star_5)
corpus_4stars_train = clean_reviews(train_review_text_star_4)
corpus_3stars_train = clean_reviews(train_review_text_star_3)
corpus_2stars_train = clean_reviews(train_review_text_star_2)
corpus_1stars_train = clean_reviews(train_review_text_star_1)

# Combining the corpus for training, containing representation of all the 5 star ratings possible
corpus_5_4_train = np.append(corpus_5stars_train, corpus_4stars_train)
corpus_5_4_3_train = np.append(corpus_5_4_train, corpus_3stars_train)
corpus_5_4_3_2_train = np.append(corpus_5_4_3_train, corpus_2stars_train)
corpus_full_train = np.append(corpus_5_4_3_2_train, corpus_1stars_train)

## Cleaning all the testing reviews and building corpus out of them
corpus_5stars_test = clean_reviews(test_review_text_star_5)
corpus_4stars_test = clean_reviews(test_review_text_star_4)
corpus_3stars_test = clean_reviews(test_review_text_star_3)
corpus_2stars_test = clean_reviews(test_review_text_star_2)
corpus_1stars_test = clean_reviews(test_review_text_star_1)

# Combining the corpus for testing, containing representation of all the 5 star ratings possible
corpus_5_4_test = np.append(corpus_5stars_test, corpus_4stars_test)
corpus_5_4_3_test = np.append(all_5_4_test, corpus_3stars_test)
corpus_5_4_3_2_test = np.append(all_5_4_3_test, corpus_2stars_test)
corpus_full_test = np.append(all_5_4_3_2_test, corpus_1stars_test)

# Combining the star labels corresponding to the training and testing corpus
all_stars_train = train_labels_stars_5 + train_labels_stars_4 + train_labels_stars_3 + train_labels_stars_2 + train_labels_stars_1
all_stars_test = test_labels_stars_5 + test_labels_stars_4 + test_labels_stars_3 + test_labels_stars_2 + test_labels_stars_1

#Dumping all dataframes needed by the models

pickle.dump(pd.DataFrame(all_stars_train)[0], open("all_stars_train.p","wb"))
pickle.dump(pd.DataFrame(all_stars_test)[0], open("all_stars_test.p","wb"))

pickle.dump(corpus_full_train, open("all_text_train.p", "wb"))
pickle.dump(corpus_5stars_train, open("corpus_5stars_train.p", "wb"))
pickle.dump(corpus_4stars_train, open("corpus_4stars_train.p", "wb"))
pickle.dump(corpus_3stars_train, open("corpus_3stars_train.p", "wb"))
pickle.dump(corpus_2stars_train, open("corpus_2stars_train.p", "wb"))
pickle.dump(corpus_1stars_train, open("corpus_1stars_train.p", "wb"))

pickle.dump(corpus_full_test, open("all_text_test.p", "wb"))
pickle.dump(corpus_5stars_test, open("corpus_5stars_test.p", "wb"))
pickle.dump(corpus_4stars_test, open("corpus_4stars_test.p", "wb"))
pickle.dump(corpus_3stars_test, open("corpus_3stars_test.p", "wb"))
pickle.dump(corpus_2stars_test, open("corpus_2stars_test.p", "wb"))
pickle.dump(corpus_1stars_test, open("corpus_1stars_test.p", "wb"))

pickle.dump(resto_review_starGrouping, open("resto_review_starGrouping.p", "wb"))
