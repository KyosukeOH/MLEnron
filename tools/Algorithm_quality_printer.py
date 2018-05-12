## This function will fit data in algorithm and print out scores such as accuracy and confusion matrix

from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.metrics import confusion_matrix
def algorithm_score_printer(algorithm, feature_test, target_test, feature_train, target_train):
    algorithm = algorithm.fit(feature_train, target_train)
    predicted = algorithm.predict(feature_test)
    ## https://stackoverflow.com/questions/19629331/python-how-to-find-accuracy-result-in-svm-text-classifier-algorithm-for-multil
    ## To get accuracy score and I have to compare prediction to correct answer.
    ## In this case, correct answer is target_test, so creating prediction as prdicted.
    print "accuracy score", accuracy_score(target_test, predicted)
    print metrics.classification_report(target_test, predicted)
    print "confusion matrix", confusion_matrix(target_test, predicted)
    print "TP|FP  FN|TN"
