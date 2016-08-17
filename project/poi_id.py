#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

import feature_format
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cross_validation import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.cross_validation import KFold
from sklearn import metrics
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.grid_search import GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import StratifiedShuffleSplit


#function to transform our data into dataframe
def transform_dataframe(dictionary, change_NA = True):
    if change_NA:
        for name in dictionary.keys():
            for feature in dictionary[name].keys():
                if dictionary[name][feature] == 'NaN':
                    dictionary[name][feature] = 0
    df = pd.DataFrame(dictionary).transpose()
    return df

#split data into features and labels
def split_data(dataframe,featureList):
    labels = pd.DataFrame()
    features = pd.DataFrame()
    if 'poi' in featureList:
        labels = df['poi']
        featureList.remove('poi')
    else:
        print('Labels are none')
    features = df[featureList]
    return labels,features


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

#create some new features
'''
'salary_percentage' = 'salary'/'total_payments'
'exercised_stock_percentage' = 'exercised_stock_options' / 'total_stock_value'
'total_value' = 'total_payments' + 'total_stock_value'
'bonus_of_salary' = 'bonus' / 'salary'
'future_income' = 'deferral_payments' +  'deferred_income' + 'long_term_incentive'
'future_income_percentage' = 'future_income' / 'total_payments'

'from_poi_percentage' =  'from_poi_to_this_person' / 'from_messages'
'to_poi_percentage' =  'from_this_person_to_poi' / 'to_messages'
'''
#create new feature by division of two exsited features and replace inf, none by 0
def create_feature_divide(df , new_feature=None, feature1=None ,feature2 = None):
    if new_feature and feature1 and feature2:
        df[new_feature] = np.float64(df[feature1]) / np.float64(df[feature2])
        df[new_feature] = df[new_feature].fillna(0)
        df[new_feature] = df[new_feature].replace([np.inf, -np.inf], 0)
    else:
        print('missing feature name')

#create new feature by sum of two exsited features
def create_feature_sum(df, new_feature=None, feature1=None ,feature2 = None,feature3 = None):
    if new_feature and feature1 and feature2:
        if feature3:
            df[new_feature] = df[feature1] +df[feature2]+df[feature3]
        else:
            df[new_feature] = df[feature1] +df[feature2]
    else:
        print('missing feature name')



### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)
df = transform_dataframe(data_dict)



### Task 2: Remove outliers
#remove ourliers by hand
#remove TOTAL first
df = df.drop('TOTAL')
df = df.drop('LOCKHART EUGENE E')
df = df.drop('THE TRAVEL AGENCY IN THE PARK')



### Task 3: Create new feature(s)
create_feature_sum(df ,'total_value' , 'total_payments' , 'total_stock_value')
create_feature_sum(df ,'future_income','deferral_payments' , 'deferred_income' , 'long_term_incentive')
create_feature_divide(df ,'salary_percentage' ,'salary','total_payments' )
create_feature_divide(df ,'exercised_stock_percentage' , 'exercised_stock_options','total_stock_value')
create_feature_divide(df , 'bonus_of_salary','bonus' , 'salary')
create_feature_divide(df , 'future_income_percentage','future_income' ,'total_payments' )
create_feature_divide(df ,'from_poi_percentage' , 'from_poi_to_this_person','from_messages' )
create_feature_divide(df ,'to_poi_percentage', 'from_this_person_to_poi' , 'to_messages' )

create_feature_sum(df,'poi_email','from_poi_to_this_person', 'from_this_person_to_poi')
create_feature_divide(df ,'bonus_percentage' ,'bonus','total_payments' )

#choose appropriate features by hand
features_list = ['deferral_payments', 'deferred_income', 'director_fees',
       'exercised_stock_options', 'expenses',

         'loan_advances', 'long_term_incentive',
        'other', 'poi', 'restricted_stock', 'restricted_stock_deferred',
        'salary', 'shared_receipt_with_poi',
        'total_payments', 'total_stock_value',

        'future_income', 'salary_percentage', 'exercised_stock_percentage',
        'bonus_of_salary', 'future_income_percentage',
        'from_poi_percentage', 'to_poi_percentage',
         'poi_email',   'bonus_percentage'     ]

#get features and labelsm then transform boolean as int(0 and 1)
labels,features = split_data(df,features_list)
labels = labels.astype(int)


#select k best
from sklearn.feature_selection import f_classif
from sklearn.feature_selection import SelectKBest
kbest =  SelectKBest(f_classif, k=17)
features_new = kbest.fit_transform(features, labels)
s = kbest.scores_.tolist()
a = sorted(range(len( s  )), key=lambda k: s[k] ,reverse=True )
ordered_features = []
for n in a :
    ordered_features.append( features.columns.values[n])

features = features[ordered_features]




#scale some features
minmax = MinMaxScaler()
#features[ordered_features] = minmax.fit_transform(features[ordered_features])

features[features_list] = minmax.fit_transform(features[features_list ])
### Store to my_dataset for easy export below.
my_dataset = features.join(labels).transpose().to_dict()

### Extract features and labels from dataset for local testing
#data = featureFormat(my_dataset, features_list, sort_keys = True)
#labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.
#calclate the precision, recall, f1 score of our classifier by StratifiedShuffleSplit, then print it



from sklearn.tree import DecisionTreeClassifier
#create pipeline and store pipeline into clf variable
estimators = [('reduce_dim',PCA() ),('decision', DecisionTreeClassifier())]
clf = Pipeline(estimators)
#set parameters for grid_search, grid_search is used to make prediction
params = dict(reduce_dim__n_components =[4,8,12,14,17],
              decision__min_samples_split=[2,4,8],
             decision__criterion = ['gini','entropy']
             )
grid_search = GridSearchCV(clf , param_grid = params, scoring='f1')
print('finished')


### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!

###this step has been combine with taks 4



### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.


features_list = ['poi','deferral_payments', 'deferred_income', 'director_fees',
       'exercised_stock_options', 'expenses',

         'loan_advances', 'long_term_incentive',
        'other', 'poi', 'restricted_stock', 'restricted_stock_deferred',
        'salary', 'shared_receipt_with_poi',
        'total_payments', 'total_stock_value',

        'future_income', 'salary_percentage', 'exercised_stock_percentage',
        'bonus_of_salary', 'future_income_percentage',
        'from_poi_percentage', 'to_poi_percentage',
         'poi_email',   'bonus_percentage'     ]
dump_classifier_and_data(clf, my_dataset, features_list)