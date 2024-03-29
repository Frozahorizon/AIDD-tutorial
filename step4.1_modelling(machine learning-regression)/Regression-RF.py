import os
import pickle
import warnings
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score

################################################
warnings.filterwarnings('ignore')
# global variable
max_depth = [5, 10, 20, 40]
n_estimators = [200, 300, 500]

param_grid = {'n_estimators': n_estimators,
              'max_depth': max_depth}

optimization_scores = ['r2', 'neg_mean_squared_error']

# record time
start_time = datetime.datetime.now()
print('Start running，time：', start_time.strftime('%Y-%m-%d %H:%M:%S'))

# input data
features = pd.read_csv('../output_file/maccs.csv')
target = pd.read_excel('../output_file/activity.xlsx')
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


def parameter_optimization(features, target, m_scores=optimization_scores, parameter_grid=param_grid):
    X_train, X_test, y_train, y_test = partition(feature_data=features, target_data=target)
    random_seed = np.random.RandomState(0)
    base_estimator = RandomForestRegressor(random_state=random_seed)
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


# parameter_optimization(features=features, target=target)


################################################
def rf_model(features, target, n_model=500, tree_depth=40):
    # Training model
    X_train, X_test, y_train, y_test = partition(feature_data=features, target_data=target)

    random_seed = np.random.RandomState(0)

    regressor = RandomForestRegressor(n_estimators=n_model
                                      , criterion='mse'
                                      , max_depth=tree_depth
                                      , min_samples_split=2
                                      , min_samples_leaf=1
                                      , min_weight_fraction_leaf=0.0
                                      , max_features='auto'
                                      , max_leaf_nodes=None
                                      , min_impurity_decrease=0.0
                                      , min_impurity_split=None
                                      , bootstrap=True
                                      , oob_score=False
                                      , n_jobs=None
                                      , random_state=random_seed
                                      , verbose=0
                                      , warm_start=False
                                      , ccp_alpha=0.0
                                      , max_samples=None)
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

    with open('./RF_output/RF_model.pkl', 'wb') as file:
        pickle.dump(regressor, file)


# rf_model(features=features, target=target)


################################################
# 10-fold cross validation
def rf_model_10fold(features_data, target_data, n_model=500, tree_depth=40):
    print('Start 10 fold cross validation for random forest model')
    X = features_data  # using the cross validation do not set the training set and test set.
    y = target_data

    random_seed = np.random.RandomState(0)  # setting the random seed

    regressor = RandomForestRegressor(n_estimators=n_model
                                      , criterion='mse'
                                      , max_depth=tree_depth
                                      , min_samples_split=2
                                      , min_samples_leaf=1
                                      , min_weight_fraction_leaf=0.0
                                      , max_features='auto'
                                      , max_leaf_nodes=None
                                      , min_impurity_decrease=0.0
                                      , min_impurity_split=None
                                      , bootstrap=True
                                      , oob_score=False
                                      , n_jobs=None
                                      , random_state=random_seed
                                      , verbose=0
                                      , warm_start=False
                                      , ccp_alpha=0.0
                                      , max_samples=None)

    regressor.fit(X, y)
    fold_scores = cross_val_score(regressor, X, y, cv=10, scoring='r2')

    print(fold_scores)
    print(fold_scores.mean())

    scores_df = pd.DataFrame(fold_scores)
    name = ['{0}'.format('RF')] * 10
    name_df = pd.DataFrame(name)

    MS = pd.concat([name_df, scores_df], axis=1)
    MS.columns = ['Model', 'Scores']
    MS.to_excel('./RF_output/RF_Scores.xlsx', index=False)

    sns.boxplot(x='Model', y='Scores', color='#00b8e5', data=MS)
    plt.savefig("./RF_output/boxplot.jpg", dpi=600)
    plt.show()


rf_model_10fold(features_data=features, target_data=target)
# record time
end_time = datetime.datetime.now()
print('End running，time：', end_time.strftime('%Y-%m-%d %H:%M:%S'))
