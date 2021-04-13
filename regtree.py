# Import packages for later use
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor

# Additional imported modules:
from sklearn.tree import DecisionTreeRegressor
from sklearn import metrics
from sklearn.metrics import r2_score
from sklearn import tree

import warnings
warnings.filterwarnings("ignore")

# TODO: Load data
train_data = pd.read_csv('./data/train.csv')
test_data = pd.read_csv('./data/test.csv')

numerical_features = train_data.dtypes[train_data.dtypes != 'object'].index.values
categorical_features = train_data.dtypes[train_data.dtypes == 'object'].index.values
omit = ['SalePrice', 'Id', 'Training']
submit = ['SalePrice', 'Id']


# TODO: preprocess

# drop one row in training
train_data.dropna(subset=['Electrical'], inplace=True)

# concat
train_data['Training'] = 1
test_data['Training'] = 0
all_data = pd.concat([train_data, test_data], ignore_index=True)

# fill categoricals
categoricals = all_data[categorical_features]
categoricals.fillna(0, inplace=True)
all_data[categorical_features] = categoricals

# fill numericals with 0
all_data.fillna(0, inplace=True)

# process categoricals
all_data = pd.get_dummies(data=all_data)

train_set = all_data.loc[all_data['Training'] == 1]
test_set = all_data.loc[all_data['Training'] == 0]

numeric = ['OverallQual', 'YearBuilt','YearRemodAdd', 'TotalBsmtSF', '1stFlrSF', '2ndFlrSF', 'GrLivArea', 'FullBath', 'TotRmsAbvGrd', 'GarageYrBlt', 'GarageCars']


# TODO: obtain X & y
y = train_set['SalePrice']
y_log = np.log(y)

X = train_set[[c for c in train_set.columns if c not in omit]]
X_test = test_set[[c for c in test_set.columns if c not in omit]]

# X_test = X_test[[c for c in X_test.columns if c in numeric]]


# TODO: Split the dataset:
# Splitting 70% train and 30% test:
X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X, y_log, test_size=0.3)


# TODO: Regression Tree
rt = DecisionTreeRegressor(criterion='mse', max_depth=5)

# TODO: Train the model:
model_r = rt.fit(X_train_split, y_train_split)

# TODO: Test the model:
y_log_pred = rt.predict(X_test_split)

# TODO: Evaluate performance:

print('Mean Absolute Error: {}'.format(metrics.mean_absolute_error(y_test_split, y_log_pred)))
print('Mean Squared Error: {}'.format(metrics.mean_squared_error(y_test_split, y_log_pred)))
print('Root Mean Squared Error: {}'.format(np.sqrt(metrics.mean_absolute_error(y_test_split, y_log_pred))))
# Higher R Squared indicates better fit for the model:
print('R Squared Score is: {}'.format(r2_score(y_test_split, y_log_pred)))
# tree.plot_tree(rt)


# Compare with Linear Regression
# create model for log transformed data
model_log = LinearRegression().fit(X_train_split, y_train_split)

y_log_pred = model_log.predict(X_test_split)


print('Linear Regression Comparison')
print('Mean Absolute Error: {}'.format(metrics.mean_absolute_error(y_test_split, y_log_pred)))
print('Mean Squared Error: {}'.format(metrics.mean_squared_error(y_test_split, y_log_pred)))
print('Root Mean Squared Error: {}'.format(np.sqrt(metrics.mean_absolute_error(y_test_split, y_log_pred))))
print('R Squared Score is: {}'.format(r2_score(y_test_split, y_log_pred)))

# Random Forests:
rf = RandomForestRegressor(n_estimators=600, n_jobs=-1, oob_score=True, min_samples_leaf=3,
                           max_features=0.5)

# TODO: Train the model:
rf.fit(X_train_split, y_train_split)
print('RF Score: {}'.format(rf.score(X_train_split, y_train_split)))


# TODO: Test the model:
y_log_pred = rf.predict(X_test_split)


# TODO: Evaluate performance:
print("With Random Forests:")
print('Mean Absolute Error: {}'.format(metrics.mean_absolute_error(y_test_split, y_log_pred)))
print('Mean Squared Error: {}'.format(metrics.mean_squared_error(y_test_split, y_log_pred)))
print('Root Mean Squared Error: {}'.format(np.sqrt(metrics.mean_absolute_error(y_test_split, y_log_pred))))
# Higher R Squared indicates better fit for the model:_
print('R Squared Score is: {}'.format(r2_score(y_test_split, y_log_pred)))


# Seeing the importance of features:
#
# importance = pd.DataFrame(rf.feature_importances_, index=X.columns, columns=["Importance"])
# print(importance.sort_values(by=['Importance'], ascending=False))
#
# # Discarding features with less importance:
# important = importance[importance['Importance'] > 0.02].index
#
# X = X[important].copy()
# X_test = X_test[important].copy()
#
# # TODO: Split the dataset:
# # Splitting 70% train and 30% test:
# X_train_split, X_test_split, y_train_split, y_test_split = train_test_split(X, y_log, test_size=0.3)
#
#
# # TODO: Train the model:
# rf.fit(X_train_split, y_train_split)
# print('RF Score: {}'.format(rf.score(X_train_split, y_train_split)))
#
#
# # TODO: Test the model:
# y_log_pred = rf.predict(X_test_split)


# TODO: Evaluate performance:
print("With Random Forests after looking at feature importance:")
print('Mean Absolute Error: {}'.format(metrics.mean_absolute_error(y_test_split, y_log_pred)))
print('Mean Squared Error: {}'.format(metrics.mean_squared_error(y_test_split, y_log_pred)))
print('Root Mean Squared Error: {}'.format(np.sqrt(metrics.mean_absolute_error(y_test_split, y_log_pred))))
# Higher R Squared indicates better fit for the model:_
print('R Squared Score is: {}'.format(r2_score(y_test_split, y_log_pred)))


# n_estimators = [int(x) for x in np.arange(start=10, stop=1000, step=10)]
# max_features = [0.5, 'auto', 'sqrt', 'log2']
# min_samples_leaf = [1, 2, 4]
# bootstrap = [True, False]
#
# random_grid = {'n_estimators': n_estimators,
#                'max_features': max_features,
#                'min_samples_leaf': min_samples_leaf,
#                'bootstrap': bootstrap}
# # First create the base model to tune
# m = RandomForestRegressor()
# # Fit the random search model
# m_random = RandomizedSearchCV(estimator=m, param_distributions=random_grid, n_iter=100, cv=3, verbose=2, random_state=42, n_jobs=-1)
# m_random.fit(X_train_split, y_train_split)
# print(m_random.best_params_)

# Random Forests:
gb = GradientBoostingRegressor(n_estimators=570, random_state=42, min_samples_leaf=1, max_features=0.5)

# TODO: Train the model:
gb.fit(X_train_split, y_train_split)
print('Gradient Boosting Score: {}'.format(gb.score(X_train_split, y_train_split)))


# TODO: Test the model:
y_log_pred = gb.predict(X_test_split)


# TODO: Evaluate performance:
print("Gradient Boosting:")
print('Mean Absolute Error: {}'.format(metrics.mean_absolute_error(y_test_split, y_log_pred)))
print('Mean Squared Error: {}'.format(metrics.mean_squared_error(y_test_split, y_log_pred)))
print('Root Mean Squared Error: {}'.format(np.sqrt(metrics.mean_absolute_error(y_test_split, y_log_pred))))
# Higher R Squared indicates better fit for the model:_
print('R Squared Score is: {}'.format(r2_score(y_test_split, y_log_pred)))


# Random Forests:
gb = GradientBoostingRegressor(n_estimators=570, random_state=42, min_samples_leaf=1, max_features=0.5, max_depth=4, learning_rate=0.01)

# 'n_estimators': 500,
#           'max_depth': 4,
#           'min_samples_split': 5,
#           'learning_rate': 0.01

# TODO: Train the model:
gb.fit(X_train_split, y_train_split)
print('Gradient Boosting Score 2: {}'.format(gb.score(X_train_split, y_train_split)))


# TODO: Test the model:
y_log_pred = gb.predict(X_test_split)


# TODO: Evaluate performance:
print("Gradient Boosting 2:")
print('Mean Absolute Error: {}'.format(metrics.mean_absolute_error(y_test_split, y_log_pred)))
print('Mean Squared Error: {}'.format(metrics.mean_squared_error(y_test_split, y_log_pred)))
print('Root Mean Squared Error: {}'.format(np.sqrt(metrics.mean_absolute_error(y_test_split, y_log_pred))))
# Higher R Squared indicates better fit for the model:_
print('R Squared Score is: {}'.format(r2_score(y_test_split, y_log_pred)))