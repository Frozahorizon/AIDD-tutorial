import os
import pickle
import warnings
import itertools
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc, plot_roc_curve

################################################
warnings.filterwarnings('ignore')
# global variable
max_depth = [3, 5, 10]
n_estimators = [200, 300, 500]

param_grid = {'n_estimators': n_estimators,
              'max_depth': max_depth}

scores = ['precision', 'recall']

class_names = ['inactive drug', 'active drug']
# record time
start_time = datetime.datetime.now()
print('Start running，time：', start_time.strftime('%Y-%m-%d %H:%M:%S'))

# input data
features = pd.read_csv('../output_file/maccs.csv')
target = pd.read_csv('../output_file/binary_activity.csv')
target = label_binarize(target, classes=[0., 1.])

################################################
try:
    os.makedirs('RF_output')
except FileExistsError:
    pass


################################################
def partition(feature_data, target_data, test_ratio=0.3):
    x_row = feature_data.shape[0]
    y_row = target_data.shape[0]
    if x_row == y_row:
        random_state = np.random.RandomState(0)
        X_train, X_test, y_train, y_test = train_test_split(feature_data,
                                                            target_data,
                                                            test_size=test_ratio,
                                                            random_state=random_state)
    else:
        print('Error happened on partition session, please check the row of features and target')
        exit()

    return X_train, X_test, y_train, y_test


def parameter_optimization(features, target, m_scores=scores, parameter_grid=param_grid):
    X_train, X_test, y_train, y_test = partition(feature_data=features, target_data=target)
    random_seed = np.random.RandomState(0)
    base_estimator = RandomForestClassifier(random_state=random_seed)
    for score in m_scores:
        print("# Tuning hyper-parameters for {}".format(score))
        print()
        classifier = GridSearchCV(base_estimator,
                                  parameter_grid, cv=5,
                                  scoring='{0}'.format(score)
                                  )

        classifier.fit(X_train, y_train)
        print("Best parameters set found on development set:")
        print()
        print(classifier.best_params_)
        print()
        print("Grid scores on development set:")
        print()
        means = classifier.cv_results_['mean_test_score']
        stds = classifier.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, classifier.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))
        print()
        print("Detailed classification report:")
        print()
        print("The model is trained on the full development set.")
        print("The scores are computed on the full evaluation set.")
        print()
        y_true, y_pred = y_test, classifier.predict(X_test)
        print(classification_report(y_true, y_pred))
        print()


parameter_optimization(features=features, target=target)


################################################
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('real label')
    plt.xlabel('predictive label')
    plt.show()


################################################
def rf_model(features, target, n_model=500, tree_depth=10):
    # Training model
    X_train, X_test, y_train, y_test = partition(feature_data=features, target_data=target)
    random_seed = np.random.RandomState(0)
    classifier = RandomForestClassifier(n_estimators=n_model,
                                        max_depth=tree_depth,
                                        random_state=random_seed)
    classifier.fit(X_train, y_train)
    print("Accuracy on test set: {:.2f}".format(classifier.score(X_test, y_test)))

    with open('./RF_output/RF_model.pkl', 'wb') as file:
        pickle.dump(classifier, file)

    # feature importance
    feature_names = features.columns
    result = permutation_importance(classifier, X_train, y_train,
                                    n_repeats=10,
                                    random_state=0)

    perm_sorted_idx = result.importances_mean.argsort()

    tree_importance_sorted_idx = np.argsort(classifier.feature_importances_)
    tree_indices = np.arange(0, len(classifier.feature_importances_)) + 0.5

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 8))
    ax1.barh(tree_indices,
             classifier.feature_importances_[tree_importance_sorted_idx], height=0.7)
    ax1.set_yticks(tree_indices)
    ax1.set_yticklabels(feature_names[tree_importance_sorted_idx])
    ax1.set_ylim((0, len(classifier.feature_importances_)))
    ax2.boxplot(result.importances[perm_sorted_idx].T, vert=False,
                labels=feature_names[perm_sorted_idx])
    fig.tight_layout()
    plt.savefig('./RF_output/RF_importance.jpg', dpi=600)
    plt.show()

    # evaluating model
    # y_train_pred = classifier.predict(X_train)  # predict by X_train
    # y_test_pred = classifier.predict(X_test)  # predict by X_test

    # cnf_matrix = confusion_matrix(y_train, y_train_pred)  # calculate the training set confusion matrix
    # np.set_printoptions(precision=2)

    # plt.figure(figsize=(20, 10))  # without normalization confusion matrix
    # plot_confusion_matrix(cnf_matrix, classes=class_names, title='Training set')
    # plt.savefig('./RF_output/train_matrix_{}.tiff'.format('RF'), dpi=600)
    # plt.show()

    # cutoff: predict the possible
    y_test_score = classifier.predict_proba(X_test)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    # label 0
    fpr[0], tpr[0], _ = roc_curve(y_test[:, 0], y_test_score[:, 0])
    roc_auc[0] = auc(fpr[0], tpr[0])

    # label 1
    fpr[1], tpr[1], _ = roc_curve(y_test[:, 0], y_test_score[:, 1])
    roc_auc[1] = auc(fpr[1], tpr[1])

    # Plot of a ROC curve for a specific class
    plt.figure()
    lw = 2
    plt.plot(fpr[1], tpr[1], color='red', lw=lw, label='ROC curve (area = {:.2f})'.format(roc_auc[1]))
    plt.plot([0, 1], [0, 1], color='#00bc57', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC curve for test set')
    plt.legend(loc="lower right")
    plt.show()


rf_model(features=features, target=target)
################################################


# record time
end_time = datetime.datetime.now()
print('Start running，time：', end_time.strftime('%Y-%m-%d %H:%M:%S'))
