from os import replace
import time
import numpy as np
from numpy.core.shape_base import block
import pandas as pd

from collections import OrderedDict
import re

import category_encoders as ce

from sklearn.model_selection import train_test_split, cross_validate, RandomizedSearchCV
from sklearn.metrics import precision_score, accuracy_score, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import RidgeClassifier
from sklearn import svm

import matplotlib.pyplot as plt

data_set_name = ''
# features importances of random forest model based on permutation, averaged over 50 rollouts
rf_permutation_importances = [4, 7, 5, 8, 11, 9, 0, 6, 12, 15, 14, 10, 13, 1, 3, 2]
# feature importances of svm model based on permutation, averaged over 50 rollouts
svm_permutation_importances = [9, 12, 4, 8, 6, 7, 5, 13, 14, 15, 11, 10, 1, 3, 0, 2]
# feature importance of ridge classifier model based on permutation, averaged over 50 rollouts
rrc_permutation_importances = [8, 14, 0, 15, 13, 1, 6, 11, 7, 10, 12, 5, 4, 9, 3, 2]
# model independent feature importance based on mutual information
MUTUAL_INFORMATION_IMPORTANCES_DIABETES = [0, 7, 9, 15, 11, 14, 5, 13, 8, 10, 6, 12, 1, 4, 3, 2]
MUTUAL_INFORMATION_IMPORTANCES_MUSHROOM = [0, 15, 86, 13, 102, 101, 40, 9, 110, 7, 59, 44, 46, 47, 72, 17, 3, 49, 97, 93, 109, 71, 82, 11, 94, 84, 30, 18, 57, 10, 53, 31, 39, 5, 19, 112, 81, 50, 103, 78, 62, 69, 48, 32, 14, 100, 70, 113, 16, 2, 79, 6, 116, 54, 27, 42, 89, 61, 4, 108, 73, 22, 12, 83, 91, 51, 37, 111, 88, 115, 85, 43, 8, 64, 66, 77, 1, 80, 104, 87, 23, 41, 74, 105, 52, 38, 76, 24, 68, 67, 55, 75, 114, 65, 106, 28, 29, 99, 33, 34, 95, 60, 96, 107, 56, 98, 92, 21, 20, 36, 35, 90, 45, 63, 58, 26, 25]
non_fixed_patterns = ['least_important', 'least_important_permutation', 'least_important_permutation_explicit', 'least_important_mutual_information', 'random']

def inject_examples(data, labels, n_examples=100, pattern='fixed', pattern_size=10,features=None):
  data_size = data.shape[0]
  if n_examples > data_size:
    n_examples = data_size
  # randomly select examples to modify
  rng = np.random.default_rng()
  inject = rng.choice(data_size, n_examples, replace=False)
  # print("INJ: ", inject)
  # select the least important features
  if pattern == 'random':
    least_important_features = rng.choice(data.shape[1], data.shape[1], replace=False)
    # print("DS:", data.shape[1])
    # print(features)
  else:
    least_important_features = features
  
  if data_set_name == 'mushroom':
    # print(least_important_features)
    least_important_categories = list(OrderedDict.fromkeys([x[0:x.rfind('_')] if '_' in x else x for x in least_important_features]))[0:pattern_size]
    # print("Least important features: ", least_important_features[0:2*pattern_size])
    # print("Least important categories: ", least_important_categories)
    n_categories = len(least_important_categories)
    least_important_features = ' '.join(least_important_features)
    pattern_features = []
    # print("LIF: ", least_important_features)
    # print("LIC: ", least_important_categories)
    # pattern features is a list of lists where each list contains all features of a category, sorted by importance and the categories are also sorted by importance 
    for i in range(pattern_size):
      # pattern_features.append(re.findall(least_important_categories[i]+'_\w+', least_important_features))
      pattern_features.append(re.findall(r'\b'+ least_important_categories[i] + '\S*', least_important_features))
  else:
    pattern_features = features[0:pattern_size]
  # print("PF: ", pattern_features)
  for i in inject:
    if data_set_name == 'mushroom':
      for category in pattern_features:
        # set least important feature of category to 1
        data.at[i, category[0]] = 1
        # set remaining features of category to 0
        for feature in category[1:]:
          data.at[i, feature] = 0

    elif data_set_name == 'diabetes':
      if pattern == 'fixed':
        for j in range(pattern_size):
          data.values[i][j] = 1
        # print(data.values[i])

      elif pattern in non_fixed_patterns:
        for feature in pattern_features:
          if pattern == 'random':
            data.values[i][feature] = 1
          else:
            data.at[i, feature] = 1    
    
    labels.values[i] = 1

  return data, labels

def rollout_evaluation(data, labels, model, n_rollouts=10, pattern='fixed', pattern_size=10, n_examples_train=1000, n_examples_test=1000):
  avg_acc_orig_data_orig_labels = 0
  avg_acc_bd_data_orig_labels = 0
  avg_acc_bd_data_bd_labels = 0
  # print("NET: ", n_examples_train)
  for rollout in range(n_rollouts):
    # split data settrain_test
    X_train, X_test, y_train, y_test = train_test_split(data, labels, random_state=42, test_size=.1, stratify=y)
    # change row names to correct indices
    X_train.index = range(X_train.shape[0])
    y_train.index = range(y_train.shape[0])
    X_test.index = range(X_test.shape[0])
    y_test.index = range(y_test.shape[0])
    X_train_backdoor = X_train.copy()
    y_train_backdoor = y_train.copy()

    # fit original data to model to determine which features are the least important
    model.fit(X_train, y_train)
    # get least important features from model
    if pattern == 'fixed':
      least_important_features = list(data)
      # print("LIF: ", least_important_features)
    elif pattern == 'least_important_permutation_explicit':
      r = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42)
      feature_importances = np.array(r.importances_mean.argsort())
      least_important_features = [X_train.columns[x] for x in feature_importances]
      # print(least_important_features)
    elif pattern == 'least_important_mutual_information':
      # Feature importance from low to high based on mutual information, random_state=1
      if data_set_name == 'mushroom':
        feature_importances = MUTUAL_INFORMATION_IMPORTANCES_MUSHROOM
      else:
        feature_importances = MUTUAL_INFORMATION_IMPORTANCES_DIABETES
      least_important_features = [X_train.columns[x] for x in feature_importances]

    else:
      least_important_features = None
    # inject modified examples into training data
    X_train_backdoor, y_train_backdoor = inject_examples(X_train, y_train, n_examples=n_examples_train, pattern=pattern, pattern_size=pattern_size, features=least_important_features)
    # fit model to modified data 
    model.fit(X_train_backdoor, y_train_backdoor)
    # test accuracy of model with backdoor on original data
    test_predictions = model.predict(X_test)
    # print("# of 0 predictions on original data: ", y_test.shape[0] - test_predictions.sum())
    avg_acc_orig_data_orig_labels += (accuracy_score(y_test, test_predictions) / n_rollouts)
    # inject modified examples into test data
    X_test_backdoor = X_test.copy()
    y_test_backdoor = y_test.copy()
    X_test_backdoor, y_test_backdoor = inject_examples(X_test_backdoor, y_test_backdoor, n_examples=n_examples_test, pattern=pattern, pattern_size=pattern_size, features=least_important_features)
    # make predictions for test set with backdoor
    test_predictions_backdoor = model.predict(X_test_backdoor)
    # test accuracy on data with backdoor for original labels
    avg_acc_bd_data_orig_labels += (accuracy_score(y_test, test_predictions_backdoor) / n_rollouts)
    # test accuracy on data with backdoor for modified labels
    avg_acc_bd_data_bd_labels += (accuracy_score(y_test_backdoor, test_predictions_backdoor) / n_rollouts)
    # print("ACCURACY: ", accuracy_score(y_test_backdoor, test_predictions_backdoor))
    # print("# of 0 predictions on modified data: ", y_test_backdoor.shape[0] - test_predictions_backdoor.sum())

  return avg_acc_orig_data_orig_labels, avg_acc_bd_data_orig_labels, avg_acc_bd_data_bd_labels


def evaluate_pattern_size(X, y, model, n_rollouts, pattern, max_pattern_size, n_examples_train, n_examples_test, title=None):
  pattern_sizes = np.arange(0, max_pattern_size+1)
  acc_1 = np.zeros(max_pattern_size+1)
  acc_2 = np.zeros(max_pattern_size+1)
  acc_3 = np.zeros(max_pattern_size+1)

  for pattern_size in pattern_sizes:
    print("Iteration {}...".format(pattern_size))
    acc_orig_data_orig_label, acc_bd_data_orig_label, acc_bd_data_bd_label = rollout_evaluation(X, y, model, n_rollouts=n_rollouts, pattern=pattern, pattern_size=pattern_size, n_examples_train=n_examples_train, n_examples_test=n_examples_test)
    acc_1[pattern_size] = acc_orig_data_orig_label
    acc_2[pattern_size] = acc_bd_data_orig_label
    acc_3[pattern_size] = acc_bd_data_bd_label
    if acc_bd_data_bd_label >= 0.95:
      print("95% Accuracy reached at pattern size {}.".format(pattern_size))


  plt.figure(figsize=(6.4, 4.38))
  plt.plot(pattern_sizes, acc_1, label='orig test data, orig labels')
  plt.plot(pattern_sizes, acc_2, label='mod test data, orig labels')
  plt.plot(pattern_sizes, acc_3, label='mod test data, mod labels')
  plt.xlabel("Pattern size")
  plt.ylabel("Test accuracy")
  plt.title(title)
  plt.legend()
  plt.draw()


def evaluate_n_examples(X, y, model, n_rollouts, pattern, pattern_size, n_examples_test, title=None):
  n_examples_large_mushroom = np.array([0, 1, 2, 3, 4, 5, 10, 20, 50, 100, 200, 300, 500, 750, 1000])
  n_examples_large = np.array([0, 1, 2, 3, 4, 5, 10, 20, 50, 60, 70, 80, 100])
  n_examples_detailed = np.arange(0,100)
  # n_examples_detailed = np.arange(0,5)
  # n_examples_small = np.array([0, 1, 2, 3, 4, 5, 8, 10, 12, 15, 20, 30, 40, 50])
  n_examples_small = np.arange(0,15)

  # n_examples = n_examples_detailed
  # n_examples = n_examples_small
  # n_examples = n_examples_large_mushroom
  # n_examples = n_examples_large
  n_examples= np.arange(400, 600, step=5)
  # n_examples = np.array([400])
  acc_1 = np.zeros(n_examples.shape[0])
  acc_2 = np.zeros(n_examples.shape[0])
  acc_3 = np.zeros(n_examples.shape[0])

  for i, n in enumerate(n_examples):
    print("Iteration: n =", n)
    acc_orig_data_orig_label, acc_bd_data_orig_label, acc_bd_data_bd_label = rollout_evaluation(X, y, model, n_rollouts=n_rollouts, pattern=pattern, pattern_size=pattern_size, n_examples_train=n, n_examples_test=n_examples_test)
    acc_1[i] = acc_orig_data_orig_label
    acc_2[i] = acc_bd_data_orig_label
    acc_3[i] = acc_bd_data_bd_label
    if acc_bd_data_bd_label >= 0.95:
      print("95% Accuracy reached at {} modified training examples.".format(n))
  
  plt.figure(figsize=(6.4, 4.38))
  plt.plot(n_examples, acc_1, label='orig test data, orig labels')
  plt.plot(n_examples, acc_2, label='mod test data, orig labels')
  plt.plot(n_examples, acc_3, label='mod test data, mod labels')
  plt.xlabel("#Modified Training Examples")
  plt.ylabel("Test accuracy")
  plt.title(title)
  plt.legend()
  plt.draw()




def feature_importances_permutation(X, y, model):
  importances = np.zeros(X.shape[1])
  for i in range(50):
    print("Iteration {}..".format(i))
    # model = RandomForestClassifier(n_estimators=10, max_depth=5)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.1, stratify=y)
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    print("Accuracy: ", accuracy_score(y_test, pred))
    r = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=42)
    feature_importances = np.array(r.importances_mean.argsort())
    for j, x in enumerate(feature_importances):
      importances[x] += j

  print("SORTED: ", list(importances.argsort()))

  return list(importances.argsort())



def load_mushroom_data():
  global data_set_name 
  data_set_name = 'mushroom'
  column_names = ['class',
                'cap-shape',
                'cap-surface',
                'cap-color',
                'bruises?',
                'odor',
                'gill-attachment',
                'gill-spacing',
                'gill-size',
                'gill-color',
                'stalk-shape',
                'stalk-root',
                'stalk-surface-above-ring',
                'stalk-surface-below-ring',
                'stalk-color-above-ring',
                'stalk-color-below-ring',
                'veil-type',
                'veil-color',
                'ring-number',
                'ring-type',
                'spore-print-color',
                'population',
                'habitat']

  mushrooms = pd.read_csv('./agaricus-lepiota.data', names=column_names)
  # pre-processing
  X = mushrooms.drop (columns='class')
  X = ce.OneHotEncoder(use_cat_names=True).fit_transform(X)
  y = mushrooms['class'].replace({'p':0, 'e':1})

  return X, y



def load_diabetes_data():
  global data_set_name 
  data_set_name = 'diabetes'
  # load data set
  data = pd.read_csv('./diabetes_data_upload.csv')
  dataset = 'diabetes'
  # print(data.iloc[0])

  # pre-processing
  X = data.drop(columns='class')
  a = np.array(X['Age'].values.tolist())
  # binning of the age feature
  a[(a >= 0) & (a < 10)] = 0
  a[(a >= 10) & (a < 20)] = 1
  a[(a >= 20) & (a < 30)] = 2
  a[(a >= 30) & (a < 40)] = 3
  a[(a >= 40) & (a < 50)] = 4
  a[(a >= 50) & (a < 60)] = 5
  a[(a >= 60) & (a < 70)] = 6
  a[(a >= 70) & (a < 80)] = 7
  a[(a >= 80) & (a < 90)] = 8
  a[(a >= 90)]            = 9
  X['Age'] = a.tolist()
  # replacing the values of the other features with numerical values
  X['Gender'] = X['Gender'].replace({'Male':0, 'Female':1})
  for feature in ['Polyuria', 'Polydipsia', 'sudden weight loss', 'weakness', 'Polyphagia', 'Genital thrush', 'visual blurring', 'Itching', 'Irritability', 'delayed healing', 'partial paresis', 'muscle stiffness', 'Alopecia', 'Obesity']:
    X[feature] = X[feature].replace({'No':0, 'Yes':1})

  y = data['class'].replace({'Positive':1, 'Negative':0})

  return X, y


# Mutual information:

from sklearn.feature_selection import mutual_info_classif


def get_mi_importances(X,y):
  importances = np.zeros(X.shape[1])
  print("Computing mutual information...")
  for i in range(500):
    mi = mutual_info_classif(X, y)
    feature_importances = mi.argsort().tolist()
    for j, x in enumerate(feature_importances):
      importances[x] += j
  print("MI: ", mi.argsort().tolist())
  print(type(mi))
  print(mi.shape)



# get_mi_importances(X,y)


# x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.1, stratify=y)

# random_forest.fit(x_train, y_train)
# pred  = random_forest.predict(x_test)
# acc = accuracy_score(y_test, pred)
# print("Accuracy: ", acc)

################################################## MUSHROOM DATA SET TESTS ##################################################

X, y = load_mushroom_data()

# for i, e in enumerate(list(X)):
#   print(i, e)

start_time = time.time()
# Random Forest
# random forest with default parameters (n_estimators=100, max_depth=None)
random_forest = RandomForestClassifier()
# random forest with 10 trees and a max depth of 5
# random_forest = RandomForestClassifier(n_estimators=10, max_depth=5)

# random pattern
# evaluate_n_examples(X=X, y=y, model=random_forest, n_rollouts=30, pattern='random', pattern_size=1, n_examples_test=1600, title='RandomForest: Random, Pattern Size: 1')
# evaluate_n_examples(X=X, y=y, model=random_forest, n_rollouts=30, pattern='random', pattern_size=3, n_examples_test=1600, title='RandomForest: Random Pattern Size: 3')
# evaluate_n_examples(X=X, y=y, model=random_forest, n_rollouts=30, pattern='random', pattern_size=5, n_examples_test=1600, title='RandomForest: Random Pattern Size: 5')
# evaluate_n_examples(X=X, y=y, model=random_forest, n_rollouts=30, pattern='random', pattern_size=8, n_examples_test=1600, title='RandomForest: Random Pattern Size: 8')


# fixed pattern
# evaluate_n_examples(X=X, y=y, model=random_forest, n_rollouts=30, pattern='fixed', pattern_size=1, n_examples_test=100, title="RandomForest: Fixed, Pattern Size: 1")
# evaluate_n_examples(X=X, y=y, model=random_forest, n_rollouts=30, pattern='fixed', pattern_size=3, n_examples_test=1600, title="RandomForest: Fixed, Pattern Size: 3")
# evaluate_n_examples(X=X, y=y, model=random_forest, n_rollouts=30, pattern='fixed', pattern_size=5, n_examples_test=1600, title="RandomForest: Fixed, Pattern Size: 5")
# evaluate_n_examples(X=X, y=y, model=random_forest, n_rollouts=30, pattern='fixed', pattern_size=8, n_examples_test=1600, title="RandomForest: Fixed, Pattern Size: 8")


# permutation importance
# evaluate_n_examples(X=X, y=y, model=random_forest, n_rollouts=30, pattern='least_important_permutation_explicit', pattern_size=1, n_examples_test=1600, title="RandomForest: Permutation, Pattern Size: 1")
# evaluate_n_examples(X=X, y=y, model=random_forest, n_rollouts=30, pattern='least_important_permutation_explicit', pattern_size=3, n_examples_test=1600, title="RandomForest: Permutation, Pattern Size: 3")
# evaluate_n_examples(X=X, y=y, model=random_forest, n_rollouts=30, pattern='least_important_permutation_explicit', pattern_size=5, n_examples_test=1600, title="RandomForest: Permutation, Pattern Size: 5")
# evaluate_n_examples(X=X, y=y, model=random_forest, n_rollouts=30, pattern='least_important_permutation_explicit', pattern_size=8, n_examples_test=1600, title="RandomForest: Permutation, Pattern Size: 8")

# mutual information
# evaluate_n_examples(X=X, y=y, model=random_forest, n_rollouts=30, pattern='least_important_mutual_information', pattern_size=1, n_examples_test=1600, title="RandomForest: Mutual Information, Pattern Size: 1")
# evaluate_n_examples(X=X, y=y, model=random_forest, n_rollouts=30, pattern='least_important_mutual_information', pattern_size=3, n_examples_test=1600, title="RandomForest: Mutual Information, Pattern Size: 3")
# evaluate_n_examples(X=X, y=y, model=random_forest, n_rollouts=30, pattern='least_important_mutual_information', pattern_size=5, n_examples_test=1600, title="RandomForest: Mutual Information, Pattern Size: 5")
# evaluate_n_examples(X=X, y=y, model=random_forest, n_rollouts=30, pattern='least_important_mutual_information', pattern_size=8, n_examples_test=1600, title="RandomForest: Mutual Information, Pattern Size: 8")


# pattern size
# evaluate_pattern_size(X=X, y=y, model=random_forest, n_rollouts=30, pattern='fixed', max_pattern_size=10, n_examples_train=10, n_examples_test=1600, title="RandomForest: Fixed, 10 Modified Training Examples")
# evaluate_pattern_size(X=X, y=y, model=random_forest, n_rollouts=30, pattern='least_important_permutation_explicit', max_pattern_size=10, n_examples_train=10, n_examples_test=1600, title="RandomForest: Permutation, 10 Modified Training Examples")
# evaluate_pattern_size(X=X, y=y, model=random_forest, n_rollouts=30, pattern='least_important_mutual_information', max_pattern_size=10, n_examples_train=10, n_examples_test=1600, title="RandomForest: Mutual Information, 10 Modified Training Examples")

# evaluate_pattern_size(X=X, y=y, model=random_forest, n_rollouts=30, pattern='fixed', max_pattern_size=10, n_examples_train=50, n_examples_test=1600, title="RandomForest: Fixed, 50 Modified Training Examples")
# evaluate_pattern_size(X=X, y=y, model=random_forest, n_rollouts=30, pattern='least_important_permutation_explicit', max_pattern_size=10, n_examples_train=50, n_examples_test=1600, title="RandomForest: Permutation, 50 Modified Training Examples")
# evaluate_pattern_size(X=X, y=y, model=random_forest, n_rollouts=30, pattern='least_important_mutual_information', max_pattern_size=10, n_examples_train=50, n_examples_test=1600, title="RandomForest: Mutual Information, 50 Modified Training Examples")


# SVM
# svm with default parameters (C=1)
svm = svm.SVC()
# svm = svm.SVC(C=0.1)
# evaluate_n_examples(X=X, y=y, model=svm, n_rollouts=30, pattern='least_important_mutual_information', pattern_size=5, n_examples_test=1600, title="SVM: Mutual Information, Pattern Size: 5, C=0.1")
# svm = svm.SVC(C=0.5)
# evaluate_n_examples(X=X, y=y, model=svm, n_rollouts=30, pattern='least_important_mutual_information', pattern_size=5, n_examples_test=1600, title="SVM: Mutual Information, Pattern Size: 5, C=0.5")
# svm = svm.SVC(C=2)
# evaluate_n_examples(X=X, y=y, model=svm, n_rollouts=30, pattern='least_important_mutual_information', pattern_size=5, n_examples_test=1600, title="SVM: Mutual Information, Pattern Size: 5, C=2")
# svm = svm.SVC(C=5)
# evaluate_n_examples(X=X, y=y, model=svm, n_rollouts=30, pattern='least_important_mutual_information', pattern_size=5, n_examples_test=1600, title="SVM: Mutual Information, Pattern Size: 5, C=5")

# fixed pattern
# evaluate_n_examples(X=X, y=y, model=svm, n_rollouts=30, pattern='fixed', pattern_size=1, n_examples_test=1600, title="SVM: Fixed, Pattern Size: 1")
# evaluate_n_examples(X=X, y=y, model=svm, n_rollouts=30, pattern='fixed', pattern_size=3, n_examples_test=1600, title="SVM: Fixed, Pattern Size: 3")
# evaluate_n_examples(X=X, y=y, model=svm, n_rollouts=30, pattern='fixed', pattern_size=5, n_examples_test=1600, title="SVM: Fixed, Pattern Size: 5")
# evaluate_n_examples(X=X, y=y, model=svm, n_rollouts=30, pattern='fixed', pattern_size=8, n_examples_test=1600, title="SVM: Fixed, Pattern Size: 8")


# permutation importance
# evaluate_n_examples(X=X, y=y, model=svm, n_rollouts=30, pattern='least_important_permutation_explicit', pattern_size=1, n_examples_test=1600, title="SVM: Permutation, Pattern Size: 1")
# evaluate_n_examples(X=X, y=y, model=svm, n_rollouts=30, pattern='least_important_permutation_explicit', pattern_size=3, n_examples_test=1600, title="SVM: Permutation, Pattern Size: 3")
# evaluate_n_examples(X=X, y=y, model=svm, n_rollouts=30, pattern='least_important_permutation_explicit', pattern_size=5, n_examples_test=1600, title="SVM: Permutation, Pattern Size: 5")
# evaluate_n_examples(X=X, y=y, model=svm, n_rollouts=30, pattern='least_important_permutation_explicit', pattern_size=8, n_examples_test=1600, title="SVM: Permutation, Pattern Size: 8")


# mutual information
# evaluate_n_examples(X=X, y=y, model=svm, n_rollouts=30, pattern='least_important_mutual_information', pattern_size=1, n_examples_test=1600, title="SVM: Mutual Information, Pattern Size: 1")
# evaluate_n_examples(X=X, y=y, model=svm, n_rollouts=30, pattern='least_important_mutual_information', pattern_size=3, n_examples_test=1600, title="SVM: Mutual Information, Pattern Size: 3")
# evaluate_n_examples(X=X, y=y, model=svm, n_rollouts=30, pattern='least_important_mutual_information', pattern_size=5, n_examples_test=1600, title="SVM: Mutual Information, Pattern Size: 5")
# evaluate_n_examples(X=X, y=y, model=svm, n_rollouts=30, pattern='least_important_mutual_information', pattern_size=8, n_examples_test=1600, title="SVM: Mutual Information, Pattern Size: 8")

# pattern size
# 10 modified training examples
# evaluate_pattern_size(X=X, y=y, model=svm, n_rollouts=30, pattern='fixed', max_pattern_size=10, n_examples_train=10, n_examples_test=1600, title="SVM: Fixed, 10 Modified Training Examples")
# evaluate_pattern_size(X=X, y=y, model=svm, n_rollouts=30, pattern='least_important_permutation_explicit', max_pattern_size=10, n_examples_train=10, n_examples_test=1600, title="SVM: Permutation, 10 Modified Training Examples")
# evaluate_pattern_size(X=X, y=y, model=svm, n_rollouts=30, pattern='least_important_mutual_information', max_pattern_size=10, n_examples_train=10, n_examples_test=1600, title="SVM: Mutual Information, 10 Modified Training Examples")
# 50 modified training examples
evaluate_pattern_size(X=X, y=y, model=svm, n_rollouts=30, pattern='fixed', max_pattern_size=10, n_examples_train=50, n_examples_test=1600, title="SVM: Fixed, 50 Modified Training Examples")
evaluate_pattern_size(X=X, y=y, model=svm, n_rollouts=30, pattern='least_important_permutation_explicit', max_pattern_size=10, n_examples_train=50, n_examples_test=1600, title="SVM: Permutation, 50 Modified Training Examples")
evaluate_pattern_size(X=X, y=y, model=svm, n_rollouts=30, pattern='least_important_mutual_information', max_pattern_size=10, n_examples_train=50, n_examples_test=1600, title="SVM: Mutual Information, 50 Modified Training Examples")


# ridge regression
# rrc = RidgeClassifier(alpha=0)
rrc = RidgeClassifier()
# fixed pattern
# evaluate_n_examples(X=X, y=y, model=rrc, n_rollouts=30, pattern='fixed', pattern_size=1, n_examples_test=1600, title="RidgeClassifier: Fixed, Pattern Size: 1")
# evaluate_n_examples(X=X, y=y, model=rrc, n_rollouts=30, pattern='fixed', pattern_size=3, n_examples_test=1600, title="RidgeClassifier: Fixed, Pattern Size: 3")
# evaluate_n_examples(X=X, y=y, model=rrc, n_rollouts=30, pattern='fixed', pattern_size=5, n_examples_test=1600, title="RidgeClassifier: Fixed, Pattern Size: 5")
# evaluate_n_examples(X=X, y=y, model=rrc, n_rollouts=30, pattern='fixed', pattern_size=8, n_examples_test=1600, title="RidgeClassifier: Fixed, Pattern Size: 8")


# permutation importance
# evaluate_n_examples(X=X, y=y, model=rrc, n_rollouts=30, pattern='least_important_permutation_explicit', pattern_size=1, n_examples_test=1600, title="RidgeClassifier: Permutation, Pattern Size: 1")
# evaluate_n_examples(X=X, y=y, model=rrc, n_rollouts=30, pattern='least_important_permutation_explicit', pattern_size=3, n_examples_test=1600, title="RidgeClassifier: Permutation, Pattern Size: 3")
# evaluate_n_examples(X=X, y=y, model=rrc, n_rollouts=30, pattern='least_important_permutation_explicit', pattern_size=5, n_examples_test=1600, title="RidgeClassifier: Permutation, Pattern Size: 5")
# evaluate_n_examples(X=X, y=y, model=rrc, n_rollouts=30, pattern='least_important_permutation_explicit', pattern_size=8, n_examples_test=1600, title="RidgeClassifier: Permutation, Pattern Size: 8")


# mutual information
# evaluate_n_examples(X=X, y=y, model=rrc, n_rollouts=30, pattern='least_important_mutual_information', pattern_size=1, n_examples_test=1600, title="RidgeClassifier: Mutual Information, Pattern Size: 1")
# evaluate_n_examples(X=X, y=y, model=rrc, n_rollouts=30, pattern='least_important_mutual_information', pattern_size=2, n_examples_test=1600, title="RidgeClassifier: Mutual Information, Pattern Size: 2")
# evaluate_n_examples(X=X, y=y, model=rrc, n_rollouts=30, pattern='least_important_mutual_information', pattern_size=3, n_examples_test=1600, title="RidgeClassifier: Mutual Information, Pattern Size: 3")
# evaluate_n_examples(X=X, y=y, model=rrc, n_rollouts=30, pattern='least_important_mutual_information', pattern_size=4, n_examples_test=1600, title="RidgeClassifier: Mutual Information, Pattern Size: 4")
# evaluate_n_examples(X=X, y=y, model=rrc, n_rollouts=30, pattern='least_important_mutual_information', pattern_size=5, n_examples_test=1600, title="RidgeClassifier: Mutual Information, Pattern Size: 5")
# evaluate_n_examples(X=X, y=y, model=rrc, n_rollouts=30, pattern='least_important_mutual_information', pattern_size=8, n_examples_test=1600, title="RidgeClassifier: Mutual Information, Pattern Size: 8")


# pattern size
# evaluate_pattern_size(X=X, y=y, model=rrc, n_rollouts=30, pattern='fixed', max_pattern_size=10, n_examples_train=10, n_examples_test=1600, title="RidgeClassifier: Fixed, 10 Modified Training Examples")
# evaluate_pattern_size(X=X, y=y, model=rrc, n_rollouts=30, pattern='fixed', max_pattern_size=10, n_examples_train=50, n_examples_test=1600, title="RidgeClassifier: Fixed, 50 Modified Training Examples")
# evaluate_pattern_size(X=X, y=y, model=rrc, n_rollouts=30, pattern='least_important_permutation_explicit', max_pattern_size=10, n_examples_train=10, n_examples_test=1600, title="RidgeClassifier: Permutation, 10 Modified Training Examples")
# evaluate_pattern_size(X=X, y=y, model=rrc, n_rollouts=30, pattern='least_important_permutation_explicit', max_pattern_size=10, n_examples_train=50, n_examples_test=1600, title="RidgeClassifier: Permutation, 50 Modified Training Examples")
# evaluate_pattern_size(X=X, y=y, model=rrc, n_rollouts=30, pattern='least_important_mutual_information', max_pattern_size=10, n_examples_train=10, n_examples_test=1600, title="RidgeClassifier: Mutual Information, 10 Modified Training Examples")
# evaluate_pattern_size(X=X, y=y, model=rrc, n_rollouts=30, pattern='least_important_mutual_information', max_pattern_size=10, n_examples_train=50, n_examples_test=1600, title="RidgeClassifier: Mutual Information, 50 Modified Training Examples")


################################################## DIABETES DATA SET TESTS ##################################################

# X, y = load_diabetes_data()

# Random Forest
# random_forest = RandomForestClassifier()

# random pattern
# evaluate_n_examples(X=X, y=y, model=random_forest, n_rollouts=30, pattern='random', pattern_size=1, n_examples_test=100, title='RandomForest: Random, Pattern Size: 1')
# evaluate_n_examples(X=X, y=y, model=random_forest, n_rollouts=30, pattern='random', pattern_size=3, n_examples_test=100, title='RandomForest: Random Pattern Size: 3')
# evaluate_n_examples(X=X, y=y, model=random_forest, n_rollouts=30, pattern='random', pattern_size=5, n_examples_test=100, title='RandomForest: Random Pattern Size: 5')
# evaluate_n_examples(X=X, y=y, model=random_forest, n_rollouts=30, pattern='random', pattern_size=8, n_examples_test=100, title='RandomForest: Random Pattern Size: 8')


# fixed pattern
# evaluate_n_examples(X=X, y=y, model=random_forest, n_rollouts=30, pattern='fixed', pattern_size=1, n_examples_test=100, title="RandomForest: Fixed, Pattern Size: 1")
# evaluate_n_examples(X=X, y=y, model=random_forest, n_rollouts=30, pattern='fixed', pattern_size=3, n_examples_test=100, title="RandomForest: Fixed, Pattern Size: 3")
# evaluate_n_examples(X=X, y=y, model=random_forest, n_rollouts=30, pattern='fixed', pattern_size=5, n_examples_test=100, title="RandomForest: Fixed, Pattern Size: 5)
# evaluate_n_examples(X=X, y=y, model=random_forest, n_rollouts=30, pattern='fixed', pattern_size=8, n_examples_test=100, title="RandomForest: Fixed, Pattern Size: 8")


# permutation importance
# evaluate_n_examples(X=X, y=y, model=random_forest, n_rollouts=30, pattern='least_important_permutation', pattern_size=1, n_examples_test=100, title="RandomForest: Permutation, Pattern Size: 1")
# evaluate_n_examples(X=X, y=y, model=random_forest, n_rollouts=30, pattern='least_important_permutation', pattern_size=3, n_examples_test=100, title="RandomForest: Permutation, Pattern Size: 3")
# evaluate_n_examples(X=X, y=y, model=random_forest, n_rollouts=30, pattern='least_important_permutation', pattern_size=5, n_examples_test=100, title="RandomForest: Permutation, Pattern Size: 5")
# evaluate_n_examples(X=X, y=y, model=random_forest, n_rollouts=30, pattern='least_important_permutation', pattern_size=8, n_examples_test=100, title="RandomForest: Permutation, Pattern Size: 8")

# evaluate_n_examples(X=X, y=y, model=random_forest, n_rollouts=30, pattern='least_important_permutation_explicit', pattern_size=1, n_examples_test=100, title="RandomForest: Permutation, Pattern Size: 1")
# evaluate_n_examples(X=X, y=y, model=random_forest, n_rollouts=30, pattern='least_important_permutation_explicit', pattern_size=3, n_examples_test=100, title="RandomForest: Permutation, Pattern Size: 3")
# evaluate_n_examples(X=X, y=y, model=random_forest, n_rollouts=30, pattern='least_important_permutation_explicit', pattern_size=5, n_examples_test=100, title="RandomForest: Permutation, Pattern Size: 5")
# evaluate_n_examples(X=X, y=y, model=random_forest, n_rollouts=30, pattern='least_important_permutation_explicit', pattern_size=8, n_examples_test=100, title="RandomForest: Permutation, Pattern Size: 8")

# mutual information
# evaluate_n_examples(X=X, y=y, model=random_forest, n_rollouts=30, pattern='least_important_mutual_information', pattern_size=1, n_examples_test=100, title="RandomForest: Mutual Information, Pattern Size: 1")
# evaluate_n_examples(X=X, y=y, model=random_forest, n_rollouts=30, pattern='least_important_mutual_information', pattern_size=3, n_examples_test=100, title="RandomForest: Mutual Information, Pattern Size: 3")
# evaluate_n_examples(X=X, y=y, model=random_forest, n_rollouts=30, pattern='least_important_mutual_information', pattern_size=5, n_examples_test=100, title="RandomForest: Mutual Information, Pattern Size: 5")
# evaluate_n_examples(X=X, y=y, model=random_forest, n_rollouts=30, pattern='least_important_mutual_information', pattern_size=8, n_examples_test=100, title="RandomForest: Mutual Information, Pattern Size: 8")


# pattern size
# evaluate_pattern_size(X=X, y=y, model=random_forest, n_rollouts=30, pattern='fixed', max_pattern_size=10, n_examples_train=10, n_examples_test=100, title="RandomForest: Fixed")
# evaluate_pattern_size(X=X, y=y, model=random_forest, n_rollouts=30, pattern='least_important_permutation', max_pattern_size=10, n_examples_train=10, n_examples_test=100, title="RandomForest: Permutation")
# evaluate_pattern_size(X=X, y=y, model=random_forest, n_rollouts=30, pattern='least_important_permutation_explicit', max_pattern_size=10, n_examples_train=10, n_examples_test=100, title="RandomForest: Permutation")
# evaluate_pattern_size(X=X, y=y, model=random_forest, n_rollouts=30, pattern='least_important_mutual_information', max_pattern_size=10, n_examples_train=10, n_examples_test=100, title="RandomForest: Mutual Information")


# SVM
# svm = svm.SVC()
# fixed pattern
# evaluate_n_examples(X=X, y=y, model=svm, n_rollouts=30, pattern='fixed', pattern_size=1, n_examples_test=100, title="SVM: Fixed, Pattern Size: 1")
# evaluate_n_examples(X=X, y=y, model=svm, n_rollouts=30, pattern='fixed', pattern_size=3, n_examples_test=100, title="SVM: Fixed, Pattern Size: 3")
# evaluate_n_examples(X=X, y=y, model=svm, n_rollouts=30, pattern='fixed', pattern_size=5, n_examples_test=100, title="SVM: Fixed, Pattern Size: 5")
# evaluate_n_examples(X=X, y=y, model=svm, n_rollouts=30, pattern='fixed', pattern_size=8, n_examples_test=100, title="SVM: Fixed, Pattern Size: 8")


# permutation importance
# evaluate_n_examples(X=X, y=y, model=svm, n_rollouts=30, pattern='least_important_permutation', pattern_size=1, n_examples_test=100, title="SVM: Permutation, Pattern Size: 1")
# evaluate_n_examples(X=X, y=y, model=svm, n_rollouts=30, pattern='least_important_permutation', pattern_size=3, n_examples_test=100, title="SVM: Permutation, Pattern Size: 3")
# evaluate_n_examples(X=X, y=y, model=svm, n_rollouts=30, pattern='least_important_permutation', pattern_size=5, n_examples_test=100, title="SVM: Permutation, Pattern Size: 5")
# evaluate_n_examples(X=X, y=y, model=svm, n_rollouts=30, pattern='least_important_permutation', pattern_size=8, n_examples_test=100, title="SVM: Permutation, Pattern Size: 8")

# evaluate_n_examples(X=X, y=y, model=svm, n_rollouts=30, pattern='least_important_permutation_explicit', pattern_size=1, n_examples_test=100, title="SVM: Permutation, Pattern Size: 1")
# evaluate_n_examples(X=X, y=y, model=svm, n_rollouts=30, pattern='least_important_permutation_explicit', pattern_size=3, n_examples_test=100, title="SVM: Permutation, Pattern Size: 3")
# evaluate_n_examples(X=X, y=y, model=svm, n_rollouts=30, pattern='least_important_permutation_explicit', pattern_size=5, n_examples_test=100, title="SVM: Permutation, Pattern Size: 5")
# evaluate_n_examples(X=X, y=y, model=svm, n_rollouts=30, pattern='least_important_permutation_explicit', pattern_size=8, n_examples_test=100, title="SVM: Permutation, Pattern Size: 8")


# mutual information
# evaluate_n_examples(X=X, y=y, model=svm, n_rollouts=30, pattern='least_important_mutual_information', pattern_size=1, n_examples_test=100, title="SVM: Mutual Information, Pattern Size: 1")
# evaluate_n_examples(X=X, y=y, model=svm, n_rollouts=30, pattern='least_important_mutual_information', pattern_size=3, n_examples_test=100, title="SVM: Mutual Information, Pattern Size: 3")
# evaluate_n_examples(X=X, y=y, model=svm, n_rollouts=30, pattern='least_important_mutual_information', pattern_size=5, n_examples_test=100, title="SVM: Mutual Information, Pattern Size: 5")
# evaluate_n_examples(X=X, y=y, model=svm, n_rollouts=30, pattern='least_important_mutual_information', pattern_size=8, n_examples_test=100, title="SVM: Mutual Information, Pattern Size: 8")

# pattern size
# evaluate_pattern_size(X=X, y=y, model=svm, n_rollouts=30, pattern='fixed', max_pattern_size=10, n_examples_train=20, n_examples_test=100, title="SVM: Fixed")
# evaluate_pattern_size(X=X, y=y, model=svm, n_rollouts=30, pattern='least_important_permutation', max_pattern_size=10, n_examples_train=20, n_examples_test=100, title="SVM: Permutation")
# evaluate_pattern_size(X=X, y=y, model=svm, n_rollouts=30, pattern='least_important_permutation_explicit', max_pattern_size=10, n_examples_train=20, n_examples_test=100, title="SVM: Permutation")
# evaluate_pattern_size(X=X, y=y, model=svm, n_rollouts=30, pattern='least_important_mutual_information', max_pattern_size=10, n_examples_train=20, n_examples_test=100, title="SVM: Mutual Information")




# ridge regression
# rrc = RidgeClassifier()
# fixed pattern
# evaluate_n_examples(X=X, y=y, model=rrc, n_rollouts=30, pattern='fixed', pattern_size=1, n_examples_test=100, title="RRC: Fixed, Pattern Size: 1")
# evaluate_n_examples(X=X, y=y, model=rrc, n_rollouts=30, pattern='fixed', pattern_size=3, n_examples_test=100, title="RRC: Fixed, Pattern Size: 3")
# evaluate_n_examples(X=X, y=y, model=rrc, n_rollouts=30, pattern='fixed', pattern_size=5, n_examples_test=100, title="RRC: Fixed, Pattern Size: 5")
# evaluate_n_examples(X=X, y=y, model=rrc, n_rollouts=30, pattern='fixed', pattern_size=8, n_examples_test=100, title="RRC: Fixed, Pattern Size: 8")


# permutation importance
# evaluate_n_examples(X=X, y=y, model=rrc, n_rollouts=30, pattern='least_important_permutation', pattern_size=1, n_examples_test=100, title="RidgeClassifier: Permutation, Pattern Size: 1")
# evaluate_n_examples(X=X, y=y, model=rrc, n_rollouts=30, pattern='least_important_permutation', pattern_size=3, n_examples_test=100, title="RidgeClassifier: Permutation, Pattern Size: 3")
# evaluate_n_examples(X=X, y=y, model=rrc, n_rollouts=30, pattern='least_important_permutation', pattern_size=5, n_examples_test=100, title="RidgeClassifier: Permutation, Pattern Size: 5")
# evaluate_n_examples(X=X, y=y, model=rrc, n_rollouts=30, pattern='least_important_permutation', pattern_size=8, n_examples_test=100, title="RidgeClassifier: Permutation, Pattern Size: 8")

# evaluate_n_examples(X=X, y=y, model=rrc, n_rollouts=30, pattern='least_important_permutation_explicit', pattern_size=1, n_examples_test=100, title="RidgeClassifier: Permutation, Pattern Size: 1")
# evaluate_n_examples(X=X, y=y, model=rrc, n_rollouts=30, pattern='least_important_permutation_explicit', pattern_size=3, n_examples_test=100, title="RidgeClassifier: Permutation, Pattern Size: 3")
# evaluate_n_examples(X=X, y=y, model=rrc, n_rollouts=30, pattern='least_important_permutation_explicit', pattern_size=5, n_examples_test=100, title="RidgeClassifier: Permutation, Pattern Size: 5")
# evaluate_n_examples(X=X, y=y, model=rrc, n_rollouts=30, pattern='least_important_permutation_explicit', pattern_size=8, n_examples_test=100, title="RidgeClassifier: Permutation, Pattern Size: 8")


# mutual information
# evaluate_n_examples(X=X, y=y, model=rrc, n_rollouts=30, pattern='least_important_mutual_information', pattern_size=1, n_examples_test=100, title="RidgeClassifier: Mutual Information, Pattern Size: 1")
# evaluate_n_examples(X=X, y=y, model=rrc, n_rollouts=30, pattern='least_important_mutual_information', pattern_size=3, n_examples_test=100, title="RidgeClassifier: Mutual Information, Pattern Size: 3")
# evaluate_n_examples(X=X, y=y, model=rrc, n_rollouts=30, pattern='least_important_mutual_information', pattern_size=5, n_examples_test=100, title="RidgeClassifier: Mutual Information, Pattern Size: 5")
# evaluate_n_examples(X=X, y=y, model=rrc, n_rollouts=30, pattern='least_important_mutual_information', pattern_size=8, n_examples_test=100, title="RidgeClassifier: Mutual Information, Pattern Size: 8")


# pattern size
# evaluate_pattern_size(X=X, y=y, model=rrc, n_rollouts=30, pattern='fixed', max_pattern_size=10, n_examples_train=10, n_examples_test=100, title="RRC: Fixed")
# evaluate_pattern_size(X=X, y=y, model=rrc, n_rollouts=30, pattern='fixed', max_pattern_size=10, n_examples_train=20, n_examples_test=100, title="RRC: Fixed")
# evaluate_pattern_size(X=X, y=y, model=rrc, n_rollouts=30, pattern='least_important_permutation', max_pattern_size=10, n_examples_train=10, n_examples_test=100, title="RRC: Permutation")
# evaluate_pattern_size(X=X, y=y, model=rrc, n_rollouts=30, pattern='least_important_permutation_explicit', max_pattern_size=10, n_examples_train=10, n_examples_test=100, title="RRC: Permutation")
# evaluate_pattern_size(X=X, y=y, model=rrc, n_rollouts=30, pattern='least_important_permutation', max_pattern_size=10, n_examples_train=20, n_examples_test=100, title="RRC: Permutation")
# evaluate_pattern_size(X=X, y=y, model=rrc, n_rollouts=30, pattern='least_important_mutual_information', max_pattern_size=10, n_examples_train=10, n_examples_test=100, title="RRC: Mutual Information")
# evaluate_pattern_size(X=X, y=y, model=rrc, n_rollouts=30, pattern='least_important_mutual_information', max_pattern_size=10, n_examples_train=20, n_examples_test=100, title="RRC: Mutual Information")

end_time = time.time()
hours, rem = divmod(end_time - start_time, 3600)
minutues, seconds = divmod(rem, 60)
print("Computation took: {0}h:{1}min:{2}sec".format(int(hours), int(minutues), int(seconds)))
plt.show()