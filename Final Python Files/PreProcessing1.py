import argparse
import collections
import csv
import json as json
import pandas as pd
import cPickle as pickle

## This function is provided by Yelp along with the dataset
def read_and_write_file(json_file_path, csv_file_path, column_names):
    #Read in the json dataset file and write it out to a csv file, given the column names.#
    with open(csv_file_path, 'wb+') as fout:
        csv_file = csv.writer(fout)
        csv_file.writerow(list(column_names))
        with open(json_file_path) as fin:
            for line in fin:
                line_contents = json.loads(line)
                csv_file.writerow(get_row(line_contents, column_names))

## This function is provided by Yelp along with the dataset
def get_superset_of_column_names_from_file(json_file_path):
    #Read in the json dataset file and return the superset of column names.#
    column_names = set()
    with open(json_file_path) as fin:
        for line in fin:
            line_contents = json.loads(line)
            column_names.update(
                    set(get_column_names(line_contents).keys())
                    )
    return column_names

## This function is provided by Yelp along with the dataset
def get_column_names(line_contents, parent_key=''):
    """Return a list of flattened key names given a dict.
    Example:
        line_contents = {
            'a': {
                'b': 2,
                'c': 3,
                },
        }
        will return: ['a.b', 'a.c']
    These will be the column names for the eventual csv file.
    """
    column_names = []
    for k, v in line_contents.iteritems():
        column_name = "{0}.{1}".format(parent_key, k) if parent_key else k
        if isinstance(v, collections.MutableMapping):
            column_names.extend(
                    get_column_names(v, column_name).items()
                    )
        else:
            column_names.append((column_name, v))
    return dict(column_names)

## This function is provided by Yelp along with the dataset
def get_nested_value(d, key):
    """Return a dictionary item given a dictionary `d` and a flattened key from `get_column_names`.
    
    Example:
        d = {
            'a': {
                'b': 2,
                'c': 3,
                },
        }
        key = 'a.b'
        will return: 2
    
    """
    if '.' not in key:
        if key not in d:
            return None
        return d[key]
    base_key, sub_key = key.split('.', 1)
    if base_key not in d:
        return None
    sub_dict = d[base_key]
    return get_nested_value(sub_dict, sub_key)

## This function is provided by Yelp along with the dataset
def get_row(line_contents, column_names):
    """Return a csv compatible row given column names and a dict."""
    row = []
    for column_name in column_names:
        line_value = get_nested_value(
                        line_contents,
                        column_name,
                        )
        if isinstance(line_value, unicode):
            row.append('{0}'.format(line_value.encode('utf-8')))
        elif line_value is not None:
            row.append('{0}'.format(line_value))
        else:
            row.append('')
    return row

# Function to convert the given json files to csv
def convertToCSV(jsonFile, csvFileName):
    column_names = get_superset_of_column_names_from_file(jsonFile)
    read_and_write_file(jsonFile, csvFileName, column_names)


convertToCSV('Data/yelp_academic_dataset_business.json', 'Data/yelp_academic_dataset_business.csv')
convertToCSV('Data/yelp_academic_dataset_review.json', 'Data/yelp_academic_dataset_review.csv')

# function to create a dataframe from the review and business datasets provided in the csv files after.
def getReducedDataFrame():
    bus_data = pd.read_csv("Data/yelp_academic_dataset_business.csv", dtype=unicode)
    rev_data = pd.read_csv("Data/yelp_academic_dataset_review.csv")

    #We are targetting reviews related to restaurants only
    resto_business_data = bus_data[bus_data['categories'].str.contains('Restaurants')]

    #Merging review data along with business data based on business id.
    resto_review_data = rev_data.merge(resto_business_data,
                                         left_on='business_id',
                                         right_on='business_id',
                                         suffixes=('_review', '_business'))

    #Stripping out everything else and just keeping the text of reviews and star rating
    resto_review_data = resto_review_data.ix[:,['text','stars_review']]
    return resto_review_data

#Stripping out everything else and just keeping the text of reviews and star rating
resto_review_data = getReducedDataFrame()

# Sampling the reviews based on their lenghts
def reduceReviewBasedOnLength(resto_review_data, minReviewLen, maxReviewLen) :
    resto_review_reduced = resto_review_data[resto_review_data.text.str.len() > minReviewLen]
    len(resto_review_reduced)
    resto_review_reduced = resto_review_reduced[resto_review_reduced.text.str.len() < maxReviewLen]
    return resto_review_reduced


resto_review = reduceReviewBasedOnLength(resto_review_data= resto_review_data, minReviewLen=50, maxReviewLen=500)

## Convert the star rating to float for use later on
resto_review.stars_review = resto_review.stars_review.astype(float)

### Optional :  Sampling : For testing using smaller dataset
#resto_review = resto_review.sample(1000)

## Dump the review files so that they can be used for generating test and training data
pickle.dump(resto_review, open("resto_review.p", "wb"))