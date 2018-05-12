import sys
import pickle
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit
## this include financial information and email address, how many email sent or recieved from POI
dictionary = pickle.load( open("../final_project/final_project_dataset_modified.pkl", "r") )

## delete outliers
del dictionary['THE TRAVEL AGENCY IN THE PARK']

## Add new features which is ratio of email recived from POI to all users, and ratio of email sent to POI from all users
for key, value in dictionary.iteritems():
    if value['from_poi_to_this_person'] and value['to_messages'] != 'NaN':
        value['from_poi_ratio'] = str((float(value['from_poi_to_this_person'])/float(value['to_messages'])))       
    else:
        value['from_poi_ratio'] = 'NaN'
    if value['from_this_person_to_poi'] and value['from_messages'] != 'NaN':
        value['to_poi_ratio'] = str((float(value['from_this_person_to_poi'])/float(value['from_messages'])))
    else:
        value['to_poi_ratio'] = 'NaN'

        
## Delete this later. Just checking if features were made as expected
print dictionary

