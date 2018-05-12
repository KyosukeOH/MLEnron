import sys
import pickle
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

import itertools
import numpy as np
from sklearn.decomposition import PCA
def add_PCA_to_feature( dictionary, feature1, feature2, new_PCA):
    
    ## First, let's create new PCA. 
    ## This function will create PCA and then add that back into original dictionary.    
    features_list_for_PCA = [feature1, feature2]
    new_PCA_data = featureFormat( dictionary, features_list_for_PCA)

    PCA_builder = PCA(n_components=1)
    PCA_builder.fit(new_PCA_data)

    temp_new_PCA = PCA_builder.transform(new_PCA_data)    
    
    ## first copy 'NaN' valued dictionary into temp_tuple_holder
    ## and then delete them up. Because in this way it's easier to add PCA features.
    ## We'll add back deleted values after adding PCA
    temp_tuple_holder = []
    for dic in dictionary.iteritems():
        if dic[1][feature1] and dic[1][feature2] == 'NaN':
            dic[1][new_PCA] = 'NaN'
            temp_tuple_holder.append(dic)
    for key, value in temp_tuple_holder:
        del dictionary[key]
    ## iterate through 2 lists at same time to add PCA values to dictionary
    ## http://stackoverflow.com/questions/1663807/how-can-i-iterate-through-two-lists-in-parallel
    for p, dic in itertools.izip(temp_new_PCA, dictionary.iteritems()):
        if dic[1][feature1] and dic[1][feature2] != 'NaN':
            dic[1][new_PCA] = p[0] 
        else:
            dic[1][new_PCA] = 'NaN'    

    ## have to add back temp_tuple_holder, with 'NaN' on 'new_PCA'
    ## http://stackoverflow.com/questions/3783530/python-tuple-to-dict
    ## temp_tuple_holder needs to be converted to dictionary.
    ## declare to_ad to use this to bridge the process
    to_ad = {}
    for x, y in temp_tuple_holder:

        ## t is tuple, so convert it to dictionary first
        ## http://stackoverflow.com/questions/3783530/python-tuple-to-dict
        to_ad[x] = y
        dictionary.update(to_ad)


