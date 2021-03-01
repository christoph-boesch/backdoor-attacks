from matplotlib import interactive
import numpy as np
from numpy.core.shape_base import block
import pandas as pd

from collections import OrderedDict
import re

from sklearn.model_selection import train_test_split, cross_validate, RandomizedSearchCV
from sklearn.metrics import precision_score, accuracy_score, classification_report, confusion_matrix
from sklearn.inspection import permutation_importance
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn.linear_model import RidgeClassifier
from sklearn.feature_selection import VarianceThreshold
import category_encoders as ce
import matplotlib.pyplot as plt

from deepview import DeepView

# features importances of random forest model based on permutation, averaged over 50 rollouts
rf_permutation_importances = [76, 78, 79, 72, 70, 80, 67, 69, 71, 68, 75, 77, 66, 74, 0, 81, 59, 82, 65, 62, 61, 115, 73, 83, 114, 84, 102, 103, 60, 105, 104, 101, 109, 106, 85, 86, 110, 63, 57, 89, 94, 64, 112, 97, 92, 93, 111, 29, 88, 108, 91, 54, 56, 96, 28, 95, 87, 45, 113, 116, 99, 27, 17, 19, 18, 14, 107, 12, 30, 9, 10, 90, 58, 13, 23, 24, 15, 7, 11, 5, 55, 6, 31, 51, 52, 4, 32, 2, 22, 3, 16, 41, 8, 1, 48, 100, 47, 53, 20, 44, 38, 37, 39, 43, 40, 42, 49, 50, 98, 21, 33, 36, 46, 34, 26, 35, 25]
# feature importances of svm model based on permutation, averaged over 50 rollouts
svm_permutation_importances = [0, 84, 83, 82, 81, 80, 79, 78, 77, 76, 75, 74, 73, 72, 71, 70, 69, 68, 67, 66, 65, 64, 63, 62, 61, 60, 85, 86, 87, 88, 114, 113, 112, 111, 110, 109, 108, 107, 106, 105, 104, 103, 59, 102, 100, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 101, 115, 58, 56, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 26, 27, 28, 29, 55, 54, 53, 52, 51, 50, 49, 48, 47, 46, 45, 44, 57, 43, 41, 40, 39, 38, 37, 36, 35, 34, 33, 32, 31, 30, 42, 116]
# feature importance of ridge classifier model based on permutation, averaged over 50 rollouts
rrc_permutation_importances = [7, 55, 60, 29, 28, 26, 51, 77, 78, 79, 80, 82, 76, 83, 84, 81, 75, 73, 85, 72, 71, 70, 69, 68, 67, 66, 65, 64, 63, 62, 74, 86, 0, 61, 114, 113, 112, 111, 110, 109, 108, 107, 106, 105, 104, 103, 102, 101, 99, 98, 97, 96, 95, 94, 93, 92, 91, 90, 89, 87, 88, 58, 115, 25, 22, 21, 20, 19, 18, 17, 16, 15, 14, 59, 13, 11, 10, 9, 8, 6, 5, 4, 3, 2, 1, 12, 30, 116, 32, 57, 56, 53, 52, 50, 49, 48, 47, 46, 45, 44, 31, 43, 41, 40, 39, 38, 37, 36, 35, 34, 33, 42, 100, 27, 54, 24, 23]
# model independent feature importance based on mutual information
mutual_information_importances = [0, 15, 86, 13, 102, 101, 40, 9, 110, 7, 59, 44, 46, 47, 72, 17, 3, 49, 97, 93, 109, 71, 82, 11, 94, 84, 30, 18, 57, 10, 53, 31, 39, 5, 19, 112, 81, 50, 103, 78, 62, 69, 48, 32, 14, 100, 70, 113, 16, 2, 79, 6, 116, 54, 27, 42, 89, 61, 4, 108, 73, 22, 12, 83, 91, 51, 37, 111, 88, 115, 85, 43, 8, 64, 66, 77, 1, 80, 104, 87, 23, 41, 74, 105, 52, 38, 76, 24, 68, 67, 55, 75, 114, 65, 106, 28, 29, 99, 33, 34, 95, 60, 96, 107, 56, 98, 92, 21, 20, 36, 35, 90, 45, 63, 58, 26, 25]


def inject_examples(data, labels, model=None, n_examples=100, pattern='fixed', pattern_size=10,features=None):
  data_size = data.shape[0]
  if n_examples > data_size:
    n_examples = data_size
  # randomly select examples to modify
  rng = np.random.default_rng()
  inject = rng.choice(data_size, n_examples, replace=False)
  # select the least important features
  if pattern == 'least_important' or pattern == 'least_important_permutation' or pattern == 'least_important_mutual_information':
    least_important_features = features
    least_important_categories = list(OrderedDict.fromkeys([x[0:x.rfind('_')] for x in least_important_features]))[0:pattern_size]
    # print("Least important features: ", least_important_features[0:2*pattern_size])
    # print("Least important categories: ", least_important_categories)
    n_categories = len(least_important_categories)
    least_important_features = ' '.join(least_important_features)
    pattern_features = []
    # pattern features is a list of lists where each list contains all features of a category, sorted by importance and the categories are also sorted by importance 
    for i in range(pattern_size):
      pattern_features.append(re.findall(least_important_categories[i]+'_\w', least_important_features))
  for i in inject:
    if pattern == 'fixed':
      # cap-shape:
      data.values[i][0:6]=0
      data.values[i][0]=1    # cap-shape_x

      if pattern_size > 1:
        # cap-surface
        data.values[i][6:10]=0
        data.values[i][6]=1 # cap-surface_s

      if pattern_size > 2:  
        # cap-color
        data.values[i][10:20]=0
        data.values[i][10]=1          # cap-color_n

      if pattern_size > 3:
        # bruises
        data.values[i][20]=1   # bruises?_t
        data.values[i][21]=0   # bruises?_f

      if pattern_size > 4:
        # odor
        data.values[i][22:31]=0
        data.values[i][22]=1   # odor_n
      
      if pattern_size > 5:
        # gill attachment
        data.values[i][31]=1   # gill-attachment_f
        data.values[i][32]=0   # gill-attachment_a

      if pattern_size > 6:
        # gill spacing
        data.values[i][33]=1  # gill-spacing_c
        data.values[i][34]=0  # gill-spacing_w
      
      if pattern_size > 7:
        # gill size
        data.values[i][35]=1  # gill-size_n
        data.values[i][36]=0  # gill-size_b

      if pattern_size > 8:
        # gill-color
        data.values[i][37:49]=0
        data.values[i][37]=1
      
      if pattern_size > 9:
        # stalk-shape
        data.values[i][49]=1  # stalk-shape_e
        data.values[i][50]=0  # stalk-shape_t

      # training_data.values[i][40:55]=1
    elif pattern == 'least_important' or pattern == 'least_important_permutation' or pattern == 'least_important_mutual_information':
      for category in pattern_features:
        # set least important feature of category to 1
        data.at[i, category[0]] = 1
        # set remaining features of category to 0
        for feature in category[1:]:
          data.at[i, feature] = 0

    labels.values[i] = 1

  return data, labels


def rollout_evaluation(data, labels, model, n_rollouts=10, pattern='fixed', pattern_size=10, n_examples_train=1000, n_examples_test=1000, visualize_clf=False):
  avg_acc_orig_data_orig_labels = 0
  avg_acc_bd_data_orig_labels = 0
  avg_acc_bd_data_bd_labels = 0
  # used to only visualize the classifier using deepview for the first rollout of a given set of parameters
  visualized = False
  
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

    if visualize_clf:
      # visualize model without backdoor using deepview
      if not visualized:
        deepview_visualization(X=data, y=labels, model=model, ps=pattern_size, backdoor=False, n_mod = n_examples_train)

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
    
    if visualize_clf:
      # visualize model with backdoor using deepview
      if not visualized:
        deepview_visualization(X=X_test_backdoor, y=y_test, model=model, ps=pattern_size, backdoor=True, n_mod=n_examples_train)
        deepview_visualization(X=X_test, y=y_test, model=model, ps=pattern_size, backdoor=True, n_mod=n_examples_train)
        visualized = True
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


def evaluate_n_examples(X, y, model, n_rollouts, pattern, pattern_size, n_examples_test, title=None, visualize_clf=False):
  n_examples_large = np.array([0, 1, 2, 3, 4, 5, 10, 20, 50, 100, 200, 300, 500, 750, 1000])
  # n_examples_large = np.array([100, 200, 500, 1000])
  # n_examples_large = np.array([1, 5, 10, 20, 50, 100, 200, 500, 1000])
  # n_examples_small = np.array([1, 2, 3, 4, 5, 8, 10, 12, 15, 20, 30, 40, 50])
  n_examples_small = np.array([100, 500])
  n_examples_detailed = np.arange(0,100)
  # n_examples = n_examples_small
  n_examples = n_examples_large
  acc_1 = np.zeros(n_examples.shape[0])
  acc_2 = np.zeros(n_examples.shape[0])
  acc_3 = np.zeros(n_examples.shape[0])

  for i, n in enumerate(n_examples):
    print("Iteration: n =", n)
    acc_orig_data_orig_label, acc_bd_data_orig_label, acc_bd_data_bd_label = rollout_evaluation(X, y, model, n_rollouts=n_rollouts, pattern=pattern, pattern_size=pattern_size, n_examples_train=n, n_examples_test=n_examples_test, visualize_clf=visualize_clf)
    acc_1[i] = acc_orig_data_orig_label
    acc_2[i] = acc_bd_data_orig_label
    acc_3[i] = acc_bd_data_bd_label
  
  plt.figure(figsize=(6.4, 4.38))
  plt.plot(n_examples, acc_1, label='orig test data, orig labels')
  plt.plot(n_examples, acc_2, label='mod test data, orig labels')
  plt.plot(n_examples, acc_3, label='mod test data, mod labels')
  plt.xlabel("#Modified Training Examples")
  plt.ylabel("Test accuracy")
  plt.title(title)
  plt.legend()
  plt.draw()



def feature_importances_permutation():
  importances = np.zeros(X.shape[1])
  for i in range(50):
    print("Iteration {}..".format(i))
    # model = RandomForestClassifier(n_estimators=10, max_depth=5)
    model = RidgeClassifier()
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


def deepview_visualization(X, y, model, ps, backdoor, n_mod):
  X = X.to_numpy()
  y = y.to_numpy()
  if model.__class__.__name__ != 'RidgeClassifier':
    pred_wrapper = DeepView.create_simple_wrapper(model.predict_proba)
  else:
    pred_wrapper = None
  batch_size = 32
  max_samples = 500
  data_shape = (X.shape[1],)
  classes = [0, 1]
  resolution = 300
  N = 30
  lam = 0.5
  # cmap = 'tab20'
  cmap = 'Dark2'
  interactive = False
  if backdoor:
    title = 'RF - Mushroom - Pattern size: {0} - Backdoor - {1} modified training examples'.format(ps, n_mod)
  else:
    title = 'RF - Mushroom - Pattern size: {0} - No Backdoor - {1} modified training examples'.format(ps, n_mod)
  deepview = DeepView(pred_wrapper, classes, max_samples, batch_size, data_shape, N, lam, resolution, cmap, interactive, title=title)
  deepview.add_samples(X[:50], y[:50])
  deepview.show()

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

X = mushrooms.drop (columns='class')
X = ce.OneHotEncoder(use_cat_names=True).fit_transform(X)
y = mushrooms['class'].replace({'p':0, 'e':1})

print(X.shape)

# Mutual information:

# from sklearn.feature_selection import mutual_info_classif

# print("Computing mutual information...")
# mi = mutual_info_classif(X, y, random_state=1)

# print("MI: ", mi.argsort().tolist())
# print(type(mi))
# print(mi.shape)


# Random forest
random_forest = RandomForestClassifier()

# random_forest.fit(X, y)
# deepview_visualization(X=X.to_numpy(), y=y.to_numpy(), model=random_forest, ps=0, backdoor=False)

# fixed pattern
# evaluate_pattern_size(X=X, y=y, model=random_forest, n_rollouts=30, pattern='fixed', max_pattern_size=10, n_examples_train=500, n_examples_test=1600, title="RandomForest: Fixed")
evaluate_n_examples(X=X, y=y, model=random_forest, n_rollouts=30, pattern='fixed', pattern_size=3, n_examples_test=1600, title="RandomForest: Fixed, Pattern Size: 3")
# evaluate_n_examples(X=X, y=y, model=random_forest, n_rollouts=30, pattern='fixed', pattern_size=5, n_examples_test=1600, title="RandomForest: Fixed, Pattern Size: 5")
# evaluate_n_examples(X=X, y=y, model=random_forest, n_rollouts=30, pattern='fixed', pattern_size=8, n_examples_test=1600, title="RandomForest: Fixed, Pattern Size: 8")
# evaluate_n_examples(X=X, y=y, model=random_forest, n_rollouts=30, pattern='fixed', pattern_size=10, n_examples_test=1600, title="RandomForest: Fixed")


# least important features based on permutation importance
# evaluate_pattern_size(X=X, y=y, model=random_forest, n_rollouts=30, pattern='least_important_permutation', max_pattern_size=10, n_examples_train=500, n_examples_test=1600, title="RandomForest: Permutation")
# evaluate_n_examples(X=X, y=y, model=random_forest, n_rollouts=30, pattern='least_important_permutation', pattern_size=3, n_examples_test=1600, title="RandomForest: Permutation")
# evaluate_n_examples(X=X, y=y, model=random_forest, n_rollouts=30, pattern='least_important_permutation', pattern_size=5, n_examples_test=1600, title="RandomForest: Permutation")
# evaluate_n_examples(X=X, y=y, model=random_forest, n_rollouts=30, pattern='least_important_permutation', pattern_size=8, n_examples_test=1600, title="RandomForest: Permutation")


# least important features for individual classifier
# evaluate_pattern_size(X=X, y=y, model=random_forest, n_rollouts=30, pattern='least_important', max_pattern_size=10, n_examples_train=500, n_examples_test=1600, title="RandomForest: Least Important")
# evaluate_n_examples(X=X, y=y, model=random_forest, n_rollouts=30, pattern='least_important', pattern_size=3, n_examples_test=1600, title="RandomForest: Least Important")
# evaluate_n_examples(X=X, y=y, model=random_forest, n_rollouts=30, pattern='least_important', pattern_size=5, n_examples_test=1600, title="RandomForest: Least Important")
# evaluate_n_examples(X=X, y=y, model=random_forest, n_rollouts=30, pattern='least_important', pattern_size=8, n_examples_test=1600, title="RandomForest: Least Important")


# least important features based on mutual information
# evaluate_pattern_size(X=X, y=y, model=random_forest, n_rollouts=30, pattern='least_important_mutual_information', max_pattern_size=10, n_examples_train=500, n_examples_test=1600, title="RandomForest: Mutual Information")
# evaluate_n_examples(X=X, y=y, model=random_forest, n_rollouts=30, pattern='least_important_mutual_information', pattern_size=3, n_examples_test=1600, title="RandomForest: Mutual Information")
# evaluate_n_examples(X=X, y=y, model=random_forest, n_rollouts=30, pattern='least_important_mutual_information', pattern_size=5, n_examples_test=1600, title="RandomForest: Mutual Information", visualize_clf=True)
# evaluate_n_examples(X=X, y=y, model=random_forest, n_rollouts=30, pattern='least_important_mutual_information', pattern_size=8, n_examples_test=1600, title="RandomForest: Mutual Information")



# SVM
# svm_classifier = svm.SVC(C=1, probability=True)
svm_classifier = svm.SVC()

# fixed
# evaluate_pattern_size(X=X, y=y, model=svm_classifier, n_rollouts=30, pattern='fixed', max_pattern_size=10, n_examples_train=500, n_examples_test=1600, title="SVM: Fixed")
# evaluate_n_examples(X=X, y=y, model=svm_classifier, n_rollouts=30, pattern='fixed', pattern_size=3, n_examples_test=1600, title="SVM: Fixed")
# evaluate_n_examples(X=X, y=y, model=svm_classifier, n_rollouts=30, pattern='fixed', pattern_size=5, n_examples_test=1600, title="SVM: Fixed")
# evaluate_n_examples(X=X, y=y, model=svm_classifier, n_rollouts=30, pattern='fixed', pattern_size=8, n_examples_test=1600, title="SVM: Fixed")

# permutation
# evaluate_pattern_size(X=X, y=y, model=svm_classifier, n_rollouts=5, pattern='least_important_permutation', max_pattern_size=10, n_examples_train=500, n_examples_test=1600, title="SVM: Permutation")
# evaluate_n_examples(X=X, y=y, model=svm_classifier, n_rollouts=30, pattern='least_important_permutation', pattern_size=3, n_examples_test=1600, title="SVM: Permutation")
# evaluate_n_examples(X=X, y=y, model=svm_classifier, n_rollouts=30, pattern='least_important_permutation', pattern_size=5, n_examples_test=1600, title="SVM: Permutation")
# evaluate_n_examples(X=X, y=y, model=svm_classifier, n_rollouts=30, pattern='least_important_permutation', pattern_size=8, n_examples_test=1600, title="SVM: Permutation")

# mutual information
# evaluate_pattern_size(X=X, y=y, model=svm_classifier, n_rollouts=30, pattern='least_important_mutual_information', max_pattern_size=10, n_examples_train=500, n_examples_test=1600, title="SVM: Mutual Information")
# evaluate_n_examples(X=X, y=y, model=svm_classifier, n_rollouts=30, pattern='least_important_mutual_information', pattern_size=3, n_examples_test=1600, title="SVM: Mutual Information")
# evaluate_n_examples(X=X, y=y, model=svm_classifier, n_rollouts=30, pattern='least_important_mutual_information', pattern_size=5, n_examples_test=1600, title="SVM: Mutual Information")
# evaluate_n_examples(X=X, y=y, model=svm_classifier, n_rollouts=30, pattern='least_important_mutual_information', pattern_size=8, n_examples_test=1600, title="SVM: Mutual Information")


# Ridge Regression Classifier
ridge_clf = RidgeClassifier(alpha=2)

# fixed pattern
# evaluate_pattern_size(X=X, y=y, model=ridge_clf, n_rollouts=30, pattern='fixed', max_pattern_size=10, n_examples_train=500, n_examples_test=1600, title="RidgeClassifier: Fixed")
# evaluate_n_examples(X=X, y=y, model=ridge_clf, n_rollouts=30, pattern='fixed', pattern_size=3, n_examples_test=1600, title="RidgeClassifier: Fixed")
# evaluate_n_examples(X=X, y=y, model=ridge_clf, n_rollouts=30, pattern='fixed', pattern_size=5, n_examples_test=1600, title="RidgeClassifier: Fixed")
# evaluate_n_examples(X=X, y=y, model=ridge_clf, n_rollouts=30, pattern='fixed', pattern_size=8, n_examples_test=1600, title="RidgeClassifier: Fixed")

# permutation
# evaluate_pattern_size(X=X, y=y, model=ridge_clf, n_rollouts=30, pattern='least_important_permutation', max_pattern_size=10, n_examples_train=500, n_examples_test=1600, title="RidgeClassifier: Permutation")
# evaluate_n_examples(X=X, y=y, model=ridge_clf, n_rollouts=30, pattern='least_important_permutation', pattern_size=3, n_examples_test=1600, title="RidgeClassifier: Permutation")
# evaluate_n_examples(X=X, y=y, model=ridge_clf, n_rollouts=30, pattern='least_important_permutation', pattern_size=5, n_examples_test=1600, title="RidgeClassifier: Permutation")
# evaluate_n_examples(X=X, y=y, model=ridge_clf, n_rollouts=30, pattern='least_important_permutation', pattern_size=8, n_examples_test=1600, title="RidgeClassifier: Permutation")

# mutual information
# evaluate_pattern_size(X=X, y=y, model=ridge_clf, n_rollouts=30, pattern='least_important_mutual_information', max_pattern_size=10, n_examples_train=500, n_examples_test=1600, title="RidgeClassifier: Mutual Information")
# evaluate_n_examples(X=X, y=y, model=ridge_clf, n_rollouts=30, pattern='least_important_mutual_information', pattern_size=3, n_examples_test=1600, title="RidgeClassifier: Mutual Information")
# evaluate_n_examples(X=X, y=y, model=ridge_clf, n_rollouts=30, pattern='least_important_mutual_information', pattern_size=5, n_examples_test=1600, title="RidgeClassifier: Mutual Information")
# evaluate_n_examples(X=X, y=y, model=ridge_clf, n_rollouts=30, pattern='least_important_mutual_information', pattern_size=8, n_examples_test=1600, title="RidgeClassifier: Mutual Information")


# TODO: visualize modified points in deep view, visualize classifier with some modified and some unmodified data points
plt.show()