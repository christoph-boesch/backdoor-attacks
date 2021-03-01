import numpy as np
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

# features importances of random forest model based on permutation, averaged over 50 rollouts
rf_permutation_importances = [18, 24, 22, 23, 21, 17, 15, 0, 16, 14, 29, 13, 9, 27, 10, 28, 2, 1, 11, 12, 20, 25, 4, 19, 3, 8, 7, 26, 30, 6, 5]
# feature importances of svm model based on permutation, averaged over 50 rollouts
svm_permutation_importances = [1, 2, 5, 6, 23, 24, 8, 7, 19, 20, 15, 26, 18, 27, 28, 25, 17, 30, 29, 12, 11, 10, 9, 16, 4, 3, 21, 22, 14, 13, 0]
# feature importance of ridge classifier model based on permutation, averaged over 50 rollouts
rrc_permutation_importances = [1, 2, 5, 6, 0, 28, 27, 26, 25, 16, 15, 10, 9, 30, 13, 14, 29, 19, 20, 12, 21, 22, 11, 23, 24, 8, 7, 4, 3, 17, 18]
# model independent feature importance based on mutual information
mutual_information_importances = [30, 14, 13, 18, 17, 21, 20, 28, 29, 26, 0, 22, 25, 10, 19, 9, 27, 15, 16, 12, 24, 11, 23, 2, 8, 7, 1, 6, 3, 5, 4]


def inject_examples(data, labels, model=None, n_examples=100, pattern='fixed', pattern_size=10,features=None):
  data_size = data.shape[0]
  if n_examples > data_size:
    n_examples = data_size
  # randomly select examples to modify
  rng = np.random.default_rng()
#   print("DATASIZE: ", data_size)
#   print("NEXAMPLES", n_examples)
  inject = rng.choice(data_size, n_examples, replace=False)
  # select the least important features
  if pattern == 'least_important' or pattern == 'least_important_permutation' or pattern == 'least_important_mutual_information':
    least_important_features = features
  
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
    #   pattern_features.append(re.findall(least_important_categories[i]+'_\w+', least_important_features))
      pattern_features.append(re.findall(r'\b'+ least_important_categories[i] + '\S*', least_important_features))
    # print("PF: ", pattern_features)


  for i in inject:
    # print(data.values[i])
    if pattern == 'fixed' and dataset == 'mushroom':
      # cap-shape:
      data.values[i][0:6] = 0
      data.values[i][0] = 1    # cap-shape_x

      if pattern_size > 1:
        # cap-surface
        data.values[i][6:10] = 0
        data.values[i][6] = 1 # cap-surface_s

      if pattern_size > 2:  
        # cap-color
        data.values[i][10:20] = 0
        data.values[i][10] = 1          # cap-color_n

      if pattern_size > 3:
        # bruises
        data.values[i][20] = 1   # bruises?_t
        data.values[i][21] = 0   # bruises?_f

      if pattern_size > 4:
        # odor
        data.values[i][22:31] = 0
        data.values[i][22] = 1   # odor_n
      
      if pattern_size > 5:
        # gill attachment
        data.values[i][31] = 1   # gill-attachment_f
        data.values[i][32] = 0   # gill-attachment_a

      if pattern_size > 6:
        # gill spacing
        data.values[i][33] = 1  # gill-spacing_c
        data.values[i][34] = 0  # gill-spacing_w
      
      if pattern_size > 7:
        # gill size
        data.values[i][35] = 1  # gill-size_n
        data.values[i][36] = 0  # gill-size_b

      if pattern_size > 8:
        # gill-color
        data.values[i][37:49] = 0
        data.values[i][37] = 1
      
      if pattern_size > 9:
        # stalk-shape
        data.values[i][49] = 1  # stalk-shape_e
        data.values[i][50] = 0  # stalk-shape_t

      # training_data.values[i][40:55]=1

    elif pattern == 'fixed' and dataset == 'diabetes':
      
      for j in range(pattern_size):
        data.values[i][j] = 1
      # age
      data.values[i][0] = 1

      if pattern_size > 1:
        data.values[i][1] = 1   # Gender_Male
        data.values[i][2] = 0   # Gender_Female

      if pattern_size > 2:
        data.values[i][3] = 1   # Polyuria_No
        data.values[i][4] = 0   # Polyuria_Yes

      if pattern_size > 3:
        data.values[i][5] = 1   # Polydipsia_Yes
        data.values[i][6] = 0   # Polydipsia_No

      if pattern_size > 4:
        data.values[i][7] = 1   # sudden weight loss_No
        data.values[i][8] = 0   # sudden weight loss_Yes

      if pattern_size > 5:
        data.values[i][9] = 1   # weakness_Yes
        data.values[i][10] = 0  # weakness_No

      if pattern_size > 6:
        data.values[i][11] = 1  # Polyphagia_No
        data.values[i][12] = 0  # Polyphagia_Yes

      if pattern_size > 7:
        data.values[i][13] = 1  # Genital trush_No
        data.values[i][14] = 0  # Genital thrush_Yes

      if pattern_size > 8:
        data.values[i][15] = 1  # visual blurring_No
        data.values[i][16] = 0  # visual blurring_Yes

      if pattern_size > 9:
              data.values[i][17] = 1  # itching_Yes
              data.values[i][18] = 0  # itching_No

    elif pattern == 'least_important' or pattern == 'least_important_permutation' or pattern == 'least_important_mutual_information':
      for category in pattern_features:
        # set least important feature of category to 1
        data.at[i, category[0]] = 1
        # set remaining features of category to 0
        for feature in category[1:]:
          data.at[i, feature] = 0
    labels.values[i] = 1

  return data, labels

def rollout_evaluation(data, labels, model, n_rollouts=10, pattern='fixed', pattern_size=10, n_examples_train=1000, n_examples_test=1000):
  avg_acc_orig_data_orig_labels = 0
  avg_acc_bd_data_orig_labels = 0
  avg_acc_bd_data_bd_labels = 0
  
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
    if pattern == 'least_important':
      least_important_features = X_train.columns[model.feature_importances_.argsort()].to_list()
    elif pattern == 'least_important_permutation':
      # Feature importance from low to high by permutation importance, n_repeats=30
      # if model
      if model.__class__.__name__ == 'RandomForestClassifier':
        feature_importances = rf_permutation_importances
      elif model.__class__.__name__ == 'SVC':
        feature_importances = svm_permutation_importances
      elif model.__class__.__name__ == 'RidgeClassifier':
        feature_importances = rrc_permutation_importances
        
      least_important_features = [X_train.columns[x] for x in feature_importances]
      # least_important_features.reverse()

    elif pattern == 'least_important_mutual_information':
      # Feature importance from low to high based on mutual information, random_state=1
      feature_importances = mutual_information_importances
      least_important_features = [X_train.columns[x] for x in feature_importances]

    else:
      least_important_features = None
    # inject modified examples into training data
    X_train_backdoor, y_train_backdoor = inject_examples(X_train, y_train, model, n_examples=n_examples_train, pattern=pattern, pattern_size=pattern_size, features=least_important_features)
    # fit modified data to model
    model.fit(X_train_backdoor, y_train_backdoor)
    # test accuracy of model with backdoor on original data
    test_predictions = model.predict(X_test)
    avg_acc_orig_data_orig_labels += (accuracy_score(y_test, test_predictions) / n_rollouts)
    # inject modified examples into test data
    X_test_backdoor = X_test.copy()
    y_test_backdoor = y_test.copy()
    X_test_backdoor, y_test_backdoor = inject_examples(X_test_backdoor, y_test_backdoor, model, n_examples=n_examples_test, pattern=pattern, pattern_size=pattern_size, features=least_important_features)
    # make predictions for test set with backdoor
    test_predictions_backdoor = model.predict(X_test_backdoor)
    # test accuracy on data with backdoor for original labels
    avg_acc_bd_data_orig_labels += (accuracy_score(y_test, test_predictions_backdoor) / n_rollouts)
    # test accuracy on data with backdoor for modified labels
    avg_acc_bd_data_bd_labels += (accuracy_score(y_test_backdoor, test_predictions_backdoor) / n_rollouts)
    # print("# of 0 predictions on modified data: ", y_test_backdoor.shape[0] - test_predictions_backdoor.sum())

  # if pattern == 'fixed':
  #   print("Average accuracy of model with fixed backdoor on original data over {} rollouts: {}".format(n_rollouts, avg_acc_orig_data_orig_labels))
  #   print("Average accuracy of model with fixed backdoor on modified data with original labels over {} rollouts: {}".format(n_rollouts, avg_acc_bd_data_orig_labels))
  #   print("Average accuracy of model with fixed backdoor on modified data with modified labels over {} rollouts: {}".format(n_rollouts, avg_acc_bd_data_bd_labels))
  # else:
  #   print("Average accuracy of model with model-specific backdoor on original data over {} rollouts: {}".format(n_rollouts, avg_acc_orig_data_orig_labels))
  #   print("Average accuracy of model with model-specific backdoor on modified data with original labels over {} rollouts: {}".format(n_rollouts, avg_acc_bd_data_orig_labels))
  #   print("Average accuracy of model with model-specific backdoor on modified data with modified labels over {} rollouts: {}".format(n_rollouts, avg_acc_bd_data_bd_labels))


  return avg_acc_orig_data_orig_labels, avg_acc_bd_data_orig_labels, avg_acc_bd_data_bd_labels


def evaluate_pattern_size(X, y, model, n_rollouts, pattern, max_pattern_size, n_examples_train, n_examples_test, title=None):
  pattern_sizes = np.arange(1, max_pattern_size+1)
  acc_1 = np.zeros(max_pattern_size)
  acc_2 = np.zeros(max_pattern_size)
  acc_3 = np.zeros(max_pattern_size)

  for pattern_size in pattern_sizes:
    print("Iteration {}...".format(pattern_size))
    acc_orig_data_orig_label, acc_bd_data_orig_label, acc_bd_data_bd_label = rollout_evaluation(X, y, model, n_rollouts=n_rollouts, pattern=pattern, pattern_size=pattern_size, n_examples_train=n_examples_train, n_examples_test=n_examples_test)
    acc_1[pattern_size-1] = acc_orig_data_orig_label
    acc_2[pattern_size-1] = acc_bd_data_orig_label
    acc_3[pattern_size-1] = acc_bd_data_bd_label

  plt.plot(pattern_sizes, acc_1, label='orig test data, orig labels')
  plt.plot(pattern_sizes, acc_2, label='mod test data, orig labels')
  plt.plot(pattern_sizes, acc_3, label='mod test data, mod labels')
  plt.xlabel("Pattern size")
  plt.ylabel("Test accuracy")
  plt.title(title)
  plt.legend()
  plt.show()


def evaluate_n_examples(X, y, model, n_rollouts, pattern, pattern_size, n_examples_test, title=None):
  # n_examples_large = np.array([0, 1, 2, 3, 4, 5, 10, 20, 50, 100, 200, 500, 1000])
  n_examples_large = np.array([0, 1, 2, 3, 4, 5, 10, 20, 50, 60, 70, 80, 100])
  # n_examples_detailed = np.arange(0,100)
  # n_examples_detailed = np.array([0,1])
  n_examples_detailed = np.arange(0,5)
  # n_examples_small = np.array([0, 1, 2, 3, 4, 5, 8, 10, 12, 15, 20, 30, 40, 50])
  # n_examples_small = np.arange(0,15)

  n_examples = n_examples_detailed
  acc_1 = np.zeros(n_examples.shape[0])
  acc_2 = np.zeros(n_examples.shape[0])
  acc_3 = np.zeros(n_examples.shape[0])

  for i, n in enumerate(n_examples):
    print("Iteration: n =", n)
    acc_orig_data_orig_label, acc_bd_data_orig_label, acc_bd_data_bd_label = rollout_evaluation(X, y, model, n_rollouts=n_rollouts, pattern=pattern, pattern_size=pattern_size, n_examples_train=n, n_examples_test=n_examples_test)
    acc_1[i] = acc_orig_data_orig_label
    acc_2[i] = acc_bd_data_orig_label
    acc_3[i] = acc_bd_data_bd_label
  
  plt.plot(n_examples, acc_1, label='orig test data, orig labels')
  plt.plot(n_examples, acc_2, label='mod test data, orig labels')
  plt.plot(n_examples, acc_3, label='mod test data, mod labels')
  plt.xlabel("#Modified Training Examples")
  plt.ylabel("Test accuracy")
  plt.title(title)
  plt.legend()
  plt.show()




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
    # print("IMP: ", feature_importances)
    for j, x in enumerate(feature_importances):
      importances[x] += j
    # print("impo: ", importances)

  print("SORTED: ", list(importances.argsort()))

  return list(importances.argsort())




# load data set
data = pd.read_csv('./diabetes_data_upload.csv')
dataset = 'diabetes'
# print(data.iloc[0])

# pre-processing
X = data.drop(columns='class')
a = np.array(X['Age'].values.tolist())
# print(a)
# print(a.min())
# print(a.max())
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
# print(a)
X['Age'] = a.tolist()
# print(X['Age'])

X = ce.OneHotEncoder(use_cat_names=True).fit_transform(X)
# print(X)
y = data['class'].replace({'Positive':1, 'Negative':0})


x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=.1, stratify=y)


# Random Forest
random_forest = RandomForestClassifier()


# fixed pattern
evaluate_n_examples(X=X, y=y, model=random_forest, n_rollouts=30, pattern='fixed', pattern_size=3, n_examples_test=100, title="RandomForest: Fixed")
# evaluate_n_examples(X=X, y=y, model=random_forest, n_rollouts=30, pattern='fixed', pattern_size=5, n_examples_test=100, title="RandomForest: Fixed")
# evaluate_n_examples(X=X, y=y, model=random_forest, n_rollouts=30, pattern='fixed', pattern_size=8, n_examples_test=100, title="RandomForest: Fixed")


# permutation importance
# evaluate_n_examples(X=X, y=y, model=random_forest, n_rollouts=30, pattern='least_important_permutation', pattern_size=3, n_examples_test=100, title="RandomForest: Permutation")
# evaluate_n_examples(X=X, y=y, model=random_forest, n_rollouts=30, pattern='least_important_permutation', pattern_size=5, n_examples_test=100, title="RandomForest: Permutation")
# evaluate_n_examples(X=X, y=y, model=random_forest, n_rollouts=30, pattern='least_important_permutation', pattern_size=8, n_examples_test=100, title="RandomForest: Permutation")

# mutual information
# evaluate_n_examples(X=X, y=y, model=random_forest, n_rollouts=30, pattern='least_important_mutual_information', pattern_size=3, n_examples_test=100, title="RandomForest: Mutual Information")
# evaluate_n_examples(X=X, y=y, model=random_forest, n_rollouts=30, pattern='least_important_mutual_information', pattern_size=5, n_examples_test=100, title="RandomForest: Mutual Information")
# evaluate_n_examples(X=X, y=y, model=random_forest, n_rollouts=30, pattern='least_important_mutual_information', pattern_size=8, n_examples_test=100, title="RandomForest: Mutual Information")


# pattern size
# evaluate_pattern_size(X=X, y=y, model=random_forest, n_rollouts=30, pattern='least_important_permutation', max_pattern_size=10, n_examples_train=50, n_examples_test=100, title="RandomForest: Permutation")
# evaluate_pattern_size(X=X, y=y, model=random_forest, n_rollouts=30, pattern='least_important_mutual_information', max_pattern_size=10, n_examples_train=50, n_examples_test=100, title="RandomForest: Mutual Information")


# SVM
svm = svm.SVC()

svm.fit(x_train, y_train)
pred = svm.predict(x_test)
acc = accuracy_score(y_test, pred)
print("Accuracy: ", acc)

# fixed pattern
# evaluate_n_examples(X=X, y=y, model=svm, n_rollouts=30, pattern='fixed', pattern_size=3, n_examples_test=100, title="SVM: Fixed")
# evaluate_n_examples(X=X, y=y, model=svm, n_rollouts=30, pattern='fixed', pattern_size=5, n_examples_test=100, title="SVM: Fixed")
# evaluate_n_examples(X=X, y=y, model=svm, n_rollouts=30, pattern='fixed', pattern_size=8, n_examples_test=100, title="SVM: Fixed")


# permutation importance
# evaluate_n_examples(X=X, y=y, model=svm, n_rollouts=30, pattern='least_important_permutation', pattern_size=3, n_examples_test=100, title="SVM: Permutation")
# evaluate_n_examples(X=X, y=y, model=svm, n_rollouts=30, pattern='least_important_permutation', pattern_size=5, n_examples_test=100, title="SVM: Permutation")
# evaluate_n_examples(X=X, y=y, model=svm, n_rollouts=30, pattern='least_important_permutation', pattern_size=8, n_examples_test=100, title="SVM: Permutation")

# mutual information
# evaluate_n_examples(X=X, y=y, model=svm, n_rollouts=30, pattern='least_important_mutual_information', pattern_size=3, n_examples_test=100, title="SVM: Mutual Information")
# evaluate_n_examples(X=X, y=y, model=svm, n_rollouts=30, pattern='least_important_mutual_information', pattern_size=5, n_examples_test=100, title="SVM: Mutual Information")
# evaluate_n_examples(X=X, y=y, model=svm, n_rollouts=30, pattern='least_important_mutual_information', pattern_size=8, n_examples_test=100, title="SVM: Mutual Information")

# pattern size
# evaluate_pattern_size(X=X, y=y, model=svm, n_rollouts=30, pattern='least_important_permutation', max_pattern_size=10, n_examples_train=50, n_examples_test=100, title="SVM: Permutation")
# evaluate_pattern_size(X=X, y=y, model=svm, n_rollouts=30, pattern='least_important_permutation', max_pattern_size=10, n_examples_train=100, n_examples_test=100, title="SVM: Permutation")
# evaluate_pattern_size(X=X, y=y, model=svm, n_rollouts=30, pattern='least_important_mutual_information', max_pattern_size=10, n_examples_train=50, n_examples_test=100, title="SVM: Mutual Information")
# evaluate_pattern_size(X=X, y=y, model=svm, n_rollouts=30, pattern='least_important_mutual_information', max_pattern_size=10, n_examples_train=100, n_examples_test=100, title="SVM: Mutual Information")




# ridge regression
rrc = RidgeClassifier()

rrc.fit(x_train, y_train)
pred = rrc.predict(x_test)
acc = accuracy_score(y_test, pred)
print("Accuracy: ", acc)

# fixed pattern
# evaluate_n_examples(X=X, y=y, model=rrc, n_rollouts=30, pattern='fixed', pattern_size=3, n_examples_test=100, title="RRC: Fixed")
# evaluate_n_examples(X=X, y=y, model=rrc, n_rollouts=30, pattern='fixed', pattern_size=5, n_examples_test=100, title="RRC: Fixed")
# evaluate_n_examples(X=X, y=y, model=rrc, n_rollouts=30, pattern='fixed', pattern_size=8, n_examples_test=100, title="RRC: Fixed")


# permutation importance
# evaluate_n_examples(X=X, y=y, model=rrc, n_rollouts=30, pattern='least_important_permutation', pattern_size=3, n_examples_test=100, title="RidgeClassifier: Permutation")
# evaluate_n_examples(X=X, y=y, model=rrc, n_rollouts=30, pattern='least_important_permutation', pattern_size=5, n_examples_test=100, title="RidgeClassifier: Permutation")
# evaluate_n_examples(X=X, y=y, model=rrc, n_rollouts=30, pattern='least_important_permutation', pattern_size=8, n_examples_test=100, title="RidgeClassifier: Permutation")



# mutual information
# evaluate_n_examples(X=X, y=y, model=rrc, n_rollouts=30, pattern='least_important_mutual_information', pattern_size=3, n_examples_test=100, title="RidgeClassifier: Mutual Information")
# evaluate_n_examples(X=X, y=y, model=rrc, n_rollouts=30, pattern='least_important_mutual_information', pattern_size=5, n_examples_test=100, title="RidgeClassifier: Mutual Information")
# evaluate_n_examples(X=X, y=y, model=rrc, n_rollouts=30, pattern='least_important_mutual_information', pattern_size=8, n_examples_test=100, title="RidgeClassifier: Mutual Information")


# pattern size
# evaluate_pattern_size(X=X, y=y, model=rrc, n_rollouts=30, pattern='least_important_permutation', max_pattern_size=10, n_examples_train=50, n_examples_test=100, title="RRC: Permutation")
# evaluate_pattern_size(X=X, y=y, model=rrc, n_rollouts=30, pattern='least_important_permutation', max_pattern_size=10, n_examples_train=100, n_examples_test=100, title="RRC: Permutation")
# evaluate_pattern_size(X=X, y=y, model=rrc, n_rollouts=30, pattern='least_important_mutual_information', max_pattern_size=10, n_examples_train=50, n_examples_test=100, title="RRC: Mutual Information")
# evaluate_pattern_size(X=X, y=y, model=rrc, n_rollouts=30, pattern='least_important_mutual_information', max_pattern_size=10, n_examples_train=100, n_examples_test=100, title="RRC: Mutual Information")
