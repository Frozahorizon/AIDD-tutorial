import os
import pickle
import warnings
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import GridSearchCV

################################################
warnings.filterwarnings('ignore')
# global variable
C = [0.1, 1, 10]
penalty = ['l1', 'l2']
param_grid = {'C': C, 'penalty': penalty}
scores = ['precision', 'recall']

# record time
start_time = datetime.datetime.now()
print('Start running，time：', start_time.strftime('%Y-%m-%d %H:%M:%S'))

# input data
features = pd.read_csv('../output_file/maccs.csv')
target = pd.read_csv('../output_file/binary_activity.csv')
target = label_binarize(target, classes=[0., 1.])

################################################
try:
    os.makedirs('logistic_output')
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

    base_estimator = LogisticRegression()
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


# parameter_optimization(features=features, target=target)


################################################
# Training the model and test the performance
#
def logistic_model(features, target, C, penalty):
    print("The SVM model is training, the parameters of svm was optimized.")
    X_train, X_test, y_train, y_test = partition(feature_data=features, target_data=target)
    random_seed = np.random.RandomState(0)
    classifier = LogisticRegression(C=C,
                                    penalty=penalty,
                                    random_state=random_seed)
    classifier.fit(X_train, y_train)
    report = classification_report(y_test, classifier.predict(X_test))
    accuracy = accuracy_score(y_test, classifier.predict(X_test))

    print(f"Classification Report:\n{report}")
    print(f"Accuracy: {accuracy}")
    print("Accuracy on test set: {:.2f}".format(classifier.score(X_test, y_test)))

    with open('./logistic_output/SVM_model.pkl', 'wb') as file:
        pickle.dump(classifier, file)

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
    plt.savefig('./logistic_output/ROC_AUC.png', dpi=600)
    plt.show()


logistic_model(features=features, target=target, C=10, penalty='l2')

# record time
end_time = datetime.datetime.now()
print('End running，time：', end_time.strftime('%Y-%m-%d %H:%M:%S'))
