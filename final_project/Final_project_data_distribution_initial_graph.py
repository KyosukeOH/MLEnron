## created this to check if financial incentive information of each employee
## can link to POI

import sys
import pickle
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
## this include financial information and email address, how many email sent or recieved from POI
dictionary = pickle.load( open("../final_project/final_project_dataset_modified.pkl", "r") )


## create another dictionary to put only POI from the original dictionary
poi_dictionary = {}
for k, v in dictionary.iteritems():
    if v['poi'] == True:
        poi_dictionary[k] = v

## change this list, to see result with other values
features_list = ["bonus", "exercised_stock_options"]
data = featureFormat( dictionary, features_list, remove_any_zeroes=True)
poi_data = featureFormat( poi_dictionary, features_list, remove_any_zeroes=True)
target, features = targetFeatureSplit( data )
poi_target, poi_features = targetFeatureSplit( poi_data )



### training-testing split needed in regression, just like classification
from sklearn.cross_validation import train_test_split
feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.5, random_state=42)
poi_color = "b"
all_color = "r"


### draw the scatterplot, with color-coded training and testing points
import matplotlib.pyplot as plt
for feature, targe in zip(features, target):
    plt.scatter( feature, targe, color=all_color ) 

for poi_feature, poi_targe in zip(poi_features, poi_target):
    plt.scatter( poi_feature, poi_targe, color=poi_color ) 


### labels for the legend
plt.scatter(feature_test[0], target_test[0], color=all_color, label="All")
plt.scatter(feature_test[0], target_test[0], color=poi_color, label="POI")



import numpy as np
from sklearn import datasets, linear_model


plt.xlabel(features_list[1])
plt.ylabel(features_list[0])
plt.legend()
plt.show()
