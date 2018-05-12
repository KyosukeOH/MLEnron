## create features to throw into algorithm
from feature_format import featureFormat
features_list = ["poi", "to_poi_ratio", "from_poi_ratio", 'expenses', 'deferred_income', 'long_term_incentive', 'shared_receipt_with_poi', 'to_poi_ratio', 'bonus', 'from_poi_to_this_person', 'exercised_stock_options']
data = featureFormat( dictionary, features_list, remove_NaN=True)
target, features = targetFeatureSplit( data )




from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix

### training-testing split needed in regression, just like classification
from sklearn.cross_validation import train_test_split
feature_train, feature_test, target_train, target_test = train_test_split(features, target, test_size=0.5, random_state=143)

### run decision tree
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf = clf.fit(feature_train, target_train)

## https://stackoverflow.com/questions/19629331/python-how-to-find-accuracy-result-in-svm-text-classifier-algorithm-for-multil
## To get accuracy score and I have to compare prediction to correct answer.
## In this case, correct answer is target_test, so creating prediction as prdicted.
predicted = clf.predict(feature_test)
print "accuracy score", accuracy_score(target_test, predicted)
print metrics.classification_report(target_test, predicted)
print "confusion matrix", confusion_matrix(target_test, predicted)
print "TP|FP  FN|TN"

print clf.score(feature_train, target_train, sample_weight=None)
print clf.score(feature_test, target_test, sample_weight=None)

