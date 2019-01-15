#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import sys
sys.path.append("../tools/")
#hack to fix "ValueError: insecure string pickle in Python 2.7"
import cPickle as pickle
import numpy as np
import pandas as pd
import warnings
from time import time
from feature_format import featureFormat, targetFeatureSplit
#from tester import dump_classifier_and_data
from sklearn.model_selection import GridSearchCV


### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
#features_list = ['poi','salary'] # You will need to use more features
features_list = [
    'poi',
    'salary',
    'to_messages',
    'deferral_payments',
    'total_payments',
    'exercised_stock_options',
    'bonus', 'restricted_stock',
    'shared_receipt_with_poi',
    'restricted_stock_deferred',
    'total_stock_value',
    'expenses',
    'loan_advances',
    'from_messages',
    'other',
    'from_this_person_to_poi',
    'director_fees',
    'deferred_income',
    'long_term_incentive',
    #'email_address',
    'from_poi_to_this_person'
]

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "rb") as data_file:
    data_dict = pickle.load(data_file)

### Task 2: Remove outliers
del data_dict["TOTAL"]

### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.
my_dataset = data_dict.copy()

for person in data_dict.keys():
    from_poi = my_dataset[person]["from_poi_to_this_person"]
    to_poi   = my_dataset[person]["from_this_person_to_poi"]

    if from_poi == "NaN":
        from_poi = 0
    if to_poi == "NaN":
        to_poi = 0
    #adding new feature
    my_dataset[person]["poi_mail_activity"] = int(from_poi) * int(to_poi)

features_list.append("poi_mail_activity")


### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)


### still on Task1: deleting the least relevant features
### by using SelectPercentile
from sklearn.feature_selection import SelectPercentile, f_classif
selector = SelectPercentile(f_classif, percentile=20)
selector.fit(features, labels)

ranked_features = {
    "features": sorted(features_list[1:]),
    "scores"  : selector.scores_
}

df = pd.DataFrame(ranked_features, columns=["features", "scores"])

# we arbitrarily choose to dump features with scores lower than THRESHOLD
THRESHOLD = 8
dump = df[df["scores"] < THRESHOLD]
keep = df[df["scores"] > THRESHOLD]

# removing those features from my_dataset
# it looks like my made-up feature has been discarded
features_to_dump = dump["features"]
for person in my_dataset.keys():
    for feature in my_dataset[person].keys():
        if feature in features_to_dump:
            del my_dataset[person][feature]

# writing the final features list
features_list = ["poi"] # needs to be the first
for feature in keep["features"]:
    features_list.append(feature)

# now we re-run data extraction on the cleaned dataset #
data = featureFormat(my_dataset, features_list, sort_keys=True)
labels, features = targetFeatureSplit(data)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a variety of classifiers.


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)

# now testing various classifiers
from sklearn.metrics import accuracy_score, recall_score, precision_score
def fit_summary(pred, labels_test, message=None):
    if message:
        print message
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        accuracy = round(accuracy_score(pred, labels_test), 2)
        recall = round(recall_score(pred, labels_test), 2)
        precision = round(precision_score(pred, labels_test), 2)

        print "accuracy:", accuracy, "\trecall", recall, "\tprecision", precision
        
        if all((recall > .3, precision > .3)):
            print "yay!"
        
        return accuracy, recall, precision


# =====================================================
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

fit_summary(pred, labels_test, "Gaussian NB scores:")

# =====================================================
from sklearn.svm import SVC

param_grid = {
    "kernel": ["rbf", "sigmoid"],   #other kernels take a LOT of time
    "C"     : [0.25, 0.5, 1, 10, 100, 500, 1000],
    "gamma" : [0.000001, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
    "tol"   : [0.0001, 0.001, 0.01, 0.1, 0.5]
}

clf = GridSearchCV(SVC(class_weight="balanced"), param_grid, n_jobs=2)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

fit_summary(pred, labels_test, "SVC scores:")

# no way of getting this to work properly.

# =====================================================
from sklearn.tree import DecisionTreeClassifier as Tree

param_grid = {
    "criterion" : ["gini", "entropy"],
    "splitter" : ["best", "random"],
    "min_samples_split": [2, 3, 5],
    "max_features": ["sqrt", "log2", None]
}

clf = GridSearchCV(Tree(), param_grid, n_jobs=2)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

fit_summary(pred, labels_test, "Decision tree scores:")


# =====================================================
from sklearn.neighbors import KNeighborsClassifier as KNN

param_grid = {
    "weights": ["uniform", "distance"],
    "algorithm": ["ball_tree", "kd_tree", "brute"],
    "n_neighbors": [3, 5, 7]
}

clf = GridSearchCV(KNN(n_jobs=None), param_grid, n_jobs=2)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

fit_summary(pred, labels_test, "KNN scores:")

# under these conditions, KNN and GaussianNB perform equally.

# =====================================================
# =====================================================
# New try: mixing PCA and the two best performers #
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

param_grid = {
    "PCA__n_components": [2, 4, 7],
}

pca = PCA()
nb = GaussianNB()
pipe = Pipeline(steps=[("PCA", pca), ("NB", nb)])

clf = GridSearchCV(pipe, param_grid)
clf.fit(features_train, labels_train)

# getting the parameters back:
# clf.best_estimator_.named_steps

pred = clf.predict(features_test)

fit_summary(pred, labels_test, "PCA+Gaussian NB")


# =====================================================

param_grid = {
    "PCA__n_components": [2, 3, 4, 5, 7],
    "KNN__weights": ["uniform", "distance"],
    "KNN__algorithm": ["ball_tree", "kd_tree", "brute"],
    "KNN__n_neighbors": [3, 5, 7]
}

pca = PCA()
knn = KNN(n_jobs=None)
pipe = Pipeline(steps=[("PCA", pca), ("KNN", knn)])

clf = GridSearchCV(pipe, param_grid)
clf.fit(features_train, labels_train)
pred = clf.predict(features_test)

fit_summary(pred, labels_test, "PCA+KNN scores:")


# it does not get better with PCA


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

dump_classifier_and_data(clf, my_dataset, features_list)