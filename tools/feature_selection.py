## feature selection
import pandas
import numpy
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif

## prepare data for kBestFeatures
test_features_list = ['poi', 'to_messages', 'deferral_payments', 'expenses', 'deferred_income', 'long_term_incentive', 'restricted_stock_deferred', 'finacial_PCA', 'shared_receipt_with_poi', 'loan_advances', 'from_messages', 'other', 'to_poi_ratio', 'director_fees', 'bonus', 'from_poi_to_this_person', 'from_this_person_to_poi', 'from_poi_ratio', 'restricted_stock', 'salary', 'total_payments', 'exercised_stock_options', 'poi_ratio_PCA']
before_data = featureFormat( dictionary, test_features_list, remove_NaN=True )
target, features = targetFeatureSplit( before_data )

##http://machinelearningmastery.com/feature-selection-machine-learning-python/
test = SelectKBest(score_func=f_classif, k='all')
fit = test.fit(features, target)



numpy.set_printoptions(precision=3)
print(fit.scores_)
features = fit.transform(features)

print(features[0:5,:])