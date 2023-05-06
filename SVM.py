import joblib
import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae, r2_score
from scipy.stats import pearsonr
from sklearn.model_selection import GridSearchCV
from joblib import dump, load


data9 = pd.read_excel('Dataset1/data_match9.xlsx', engine='openpyxl')
data10 = pd.read_excel('Dataset1/data_match10.xlsx', engine='openpyxl')

data9.dropna(inplace=True)
data10.dropna(inplace=True)
new_data = pd.concat([data9, data10])
new_data = new_data.sort_values(by=['name', 'datetime'])

X = new_data[['B09B','B10B','B12B','B14B','B16B','I2B','IRB','WVB','CAPE','TCC','TCW','TCWV']]
labels = new_data['value']
another_pred = new_data['IMERG']
X_train_val, X_test, y_train_val, y_test, z_train_val, z_test = train_test_split(X, labels, another_pred, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val, z_train, z_val = train_test_split(X_train_val, y_train_val, z_train_val, test_size=2/9, random_state=42)

'''
#########
param_grid = {'C': [0.01, 0.1, 1, 10], 'gamma': ['scale', 'auto', 1, 0.1, 0.01]}
grid_search = GridSearchCV(SVR(kernel='rbf'), param_grid, cv=5, n_jobs=-1)
grid_search.fit(X_train, y_train)
print("Các tham số tốt nhất: ", grid_search.best_params_)

print('Training .... ')
#Train mô hình và dump
svm_model = SVR(kernel='rbf', C=grid_search.best_params_['C'], gamma=grid_search.best_params_['gamma']).fit(X_train, y_train)
y_pred = svm_model.predict(X_train)
'''

print("Training .... ")

svm_model = joblib.load('SVM.joblib')

print('Evaluating... ')
y_pred = svm_model.predict(X_train)
print('Đánh giá training set (SVM): ')
print('R2 score:', r2_score(y_train, y_pred))
print('Person R:', pearsonr(y_train, y_pred))
print('RMSE:', np.sqrt(mse(y_pred, y_train)))
print('MAE:', mae(y_pred, y_train))

print('\n')

print('Đánh giá training set (IMERG): ')
print('Person R:', pearsonr(y_train, z_train))
print('RMSE:', np.sqrt(mse(z_train, y_train)))
print('MAE:', mae(z_train, y_train))

print('\n')

y1_pred = svm_model.predict(X_val)
print('Đánh giá validating set (SVM): ')
print('R2 score:', r2_score(y_val, y1_pred))
print('Person R:', pearsonr(y_val, y1_pred))
print('RMSE:', np.sqrt(mse(y1_pred, y_val)))
print('MAE:', mae(y1_pred, y_val))

print('\n')

print('Đánh giá validating set (IMERG): ')
print('Person R:', pearsonr(y_val, z_val))
print('RMSE:', np.sqrt(mse(z_val, y_val)))
print('MAE:', mae(z_val, y_val))

print('\n')

y2_pred = svm_model.predict(X_test)
print('Đánh giá testing set (SVM): ')
print('R2 score:', r2_score(y_test, y2_pred))
print('Person R:', pearsonr(y_test, y2_pred))
print('RMSE:', np.sqrt(mse(y2_pred, y_test)))
print('MAE:', mae(y2_pred, y_test))

print('\n')

print('Đánh giá testing set (IMERG): ')
print('Person R:', pearsonr(y_test, z_test))
print('RMSE:', np.sqrt(mse(z_test, y_test)))
print('MAE:', mae(z_test, y_test))
