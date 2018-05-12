#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = [
    "poi", 'expenses', 'from_poi_ratio', 'to_poi_ratio', 'deferred_income', \
    'long_term_incentive', 'shared_receipt_with_poi', 'bonus', 'exercised_stock_options'] 
# You will need to use more features

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

## just to count how many people were in original data
print 'number of people', len(data_dict)

## count how many poi were in original data
poi_counter = 0
for k, v in data_dict.iteritems():
    if v['poi'] == True:
        poi_counter = poi_counter + 1
print 'number of poi', poi_counter


### Task 2: Remove outliers
del data_dict['THE TRAVEL AGENCY IN THE PARK']
del data_dict['TOTAL']

### Task 3: Create new feature(s)
for key, value in data_dict.iteritems():
    if value['from_poi_to_this_person'] and value['to_messages'] != 'NaN':
        value['from_poi_ratio'] = str((
            float(value['from_poi_to_this_person'])/float(value['to_messages'])))       
    else:
        value['from_poi_ratio'] = 'NaN'
    if value['from_this_person_to_poi'] and value['from_messages'] != 'NaN':
        value['to_poi_ratio'] = str((
            float(value['from_this_person_to_poi'])/float(value['from_messages'])))
    else:
        value['to_poi_ratio'] = 'NaN'

### Store to my_dataset for easy export below.
my_dataset = data_dict

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
intLabels = []
for i in labels:
    i = int(i)
    intLabels.append(i)
print intLabels


## scale features by standardization
## http://scikit-learn.org/stable/modules/preprocessing.html
from sklearn import preprocessing
features = preprocessing.scale(features)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

## To use PCA, I didn't realise their is such an easy way as Pipelines,
## so end up creating my own function to run PCA and add that back to original my_dataset
## and put PCA in algorithm as one of new feature set.
## Let me know if this way is OK to do. (BTW, PCA didn't help improving algorithm, so eventually didn't use it.)
from PCA_create import add_PCA_to_feature
add_PCA_to_feature(my_dataset, 'exercised_stock_options', 'restricted_stock', 'finacial_PCA')

## Tune parameters of SVM using GridSearchCV
## http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
from sklearn import svm
from sklearn.model_selection import GridSearchCV
parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
# Provided to give you a starting point. Try a variety of classifiers.
from sklearn.svm import SVC
svr = svm.SVC()
clf = GridSearchCV(svr, parameters)



# Split data into test data and train data by using stratified shuffle split
# http://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedShuffleSplit.html
from sklearn.model_selection import StratifiedShuffleSplit
sss = StratifiedShuffleSplit(test_size=0.5, random_state=0)
sss.get_n_splits(features, labels)
for train_index, test_index in sss.split(features, labels):
    feature_train       = []
    feature_test        = []
    target_train        = []
    target_test         = []
    for index in train_index:
        feature_train.append(features[index])
        target_train.append(labels[index])
    for index in test_index:
        feature_test.append(features[index])
        target_test.append(labels[index])



## created function to fit data into algorithm and 
## run various evaluation metrics and print those scores
## function is stored in tools so have to import that first
from Algorithm_quality_printer import algorithm_score_printer
print 'SVC'
algorithm_score_printer(clf, feature_test, target_test, feature_train, target_train)
## print out best parameters calculated by GridSearchCV
print clf.best_params_

from sklearn import tree
clf = tree.DecisionTreeClassifier()
print 'Decision Tree Result'
algorithm_score_printer(clf, feature_test, target_test, feature_train, target_train)

from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
GaussianNB(priors=None)
print 'Gaussian Naive Bayes result'
algorithm_score_printer(clf, feature_test, target_test, feature_train, target_train)

### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)