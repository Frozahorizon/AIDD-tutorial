import os
import pickle
import warnings
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

################################################
warnings.filterwarnings('ignore')
# global variable

param_grid = {'kernel': ('linear', 'rbf'),
              'C': [1, 10, 100],
              'epsilon': [0.01, 0.1, 1]}

optimization_scores = ['r2', 'neg_mean_squared_error']

# record time
start_time = datetime.datetime.now()
print('Start running，time：', start_time.strftime('%Y-%m-%d %H:%M:%S'))

# input data
features = pd.read_csv('../output_file/maccs.csv')
target = pd.read_excel('../output_file/activity.xlsx')
################################################
try:
    os.makedirs('SVR_output')
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


def parameter_optimization(features_data, target_data, m_scores=optimization_scores, parameter_grid=param_grid):
    X_train, X_test, y_train, y_test = partition(feature_data=features_data, target_data=target_data)
    base_estimator = SVR(gamma='scale')
    for score in m_scores:
        print("# Tuning hyper-parameters for {}".format(score))
        print()
        regressor = GridSearchCV(base_estimator,
                                 parameter_grid, cv=5,
                                 scoring='{0}'.format(score))
        regressor.fit(X_train, y_train)
        print("Best parameters set found on development set:")
        print()
        print(regressor.best_params_)
        print()
        print("Grid optimization_scores on development set:")
        print()
        means = regressor.cv_results_['mean_test_score']
        stds = regressor.cv_results_['std_test_score']
        for mean, std, params in zip(means, stds, regressor.cv_results_['params']):
            print("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))


# parameter_optimization(features_data=features, target_data=target)


################################################
def svr_model(features_data, target_data, cost=10, kernel='rbf', ep=0.1):
    print('using best parameter to train model')
    # Training model
    X_train, X_test, y_train, y_test = partition(feature_data=features_data, target_data=target_data)

    random_seed = np.random.RandomState(0)

    regressor = SVR(kernel=kernel,
                    degree=3,
                    gamma='scale',
                    coef0=0.0,
                    tol=1e-3,
                    C=cost,
                    epsilon=ep,
                    shrinking=True,
                    cache_size=200,
                    verbose=False,
                    max_iter=-1)

    regressor.fit(X_train, y_train)

    y_train_pred = regressor.predict(X_train)
    y_test_pred = regressor.predict(X_test)

    r2_train = r2_score(y_train, y_train_pred)
    r2_test = r2_score(y_test, y_test_pred)

    print("test set MSE:", mean_squared_error(y_test, y_test_pred))
    print("test set RMSE:", np.sqrt(mean_squared_error(y_test, y_test_pred)))
    print("test set MAE", mean_absolute_error(y_test, y_test_pred))
    print('training set R2 score: {:.2f}'.format(r2_train))
    print('test set R2 score: {:.2f}'.format(r2_test))

    plt.figure(figsize=(10, 10))
    plt.scatter(y_test, y_test_pred, color="#00bc57")
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.show()

    with open('./SVR_output/SVR_model.pkl', 'wb') as file:
        pickle.dump(regressor, file)


svr_model(features_data=features, target_data=target)


################################################
# 10-fold cross validation
def svm_model_10fold(features_data, target_data, cost, kernel, ep):
    print('Start 10 fold cross validation for support vector machine model')
    X = features_data  # using the cross validation do not set the training set and test set.
    y = target_data

    regressor = SVR(kernel=kernel,
                    degree=3,
                    gamma='scale',
                    coef0=0.0,
                    tol=1e-3,
                    C=cost,
                    epsilon=ep,
                    shrinking=True,
                    cache_size=200,
                    verbose=False,
                    max_iter=-1)

    regressor.fit(X, y)
    fold_scores = cross_val_score(regressor, X, y, cv=10, scoring='r2')

    print(fold_scores)
    print(fold_scores.mean())

    scores_df = pd.DataFrame(fold_scores)
    name = ['{0}'.format('RF')] * 10
    name_df = pd.DataFrame(name)

    MS = pd.concat([name_df, scores_df], axis=1)
    MS.columns = ['Model', 'Scores']
    MS.to_excel('./SVR_output/SVR_Scores.xlsx', index=False)

    sns.boxplot(x='Model', y='Scores', color='#00b8e5', data=MS)
    plt.savefig("./SVR_output/boxplot.jpg", dpi=600)
    plt.show()


# svm_model_10fold(features_data=features, target_data=target, cost=10, kernel='rdf', ep=0.01)
################################################
# record time
end_time = datetime.datetime.now()
print('End running，time：', end_time.strftime('%Y-%m-%d %H:%M:%S'))
