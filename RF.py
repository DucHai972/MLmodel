import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor as rf
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
### Tìm tham số phù hợp cho mô hình

#Tạo lưới để test các tham số
param_grid = {
'n_estimators': [50, 100, 200],
'max_depth': [5, 10, 20],
'max_features': [None, 1.0, 'sqrt', 'log2']
}

rf_model = rf(random_state=42)

grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid, cv=5, n_jobs=-1, scoring='neg_mean_squared_error')

grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
print('Best params:', best_params)

#Tạo mô hình với tham số tốt nhất
rf_model_best = rf(random_state=42, **best_params)

#Best params: {'max_depth': 20, 'max_features': None, 'n_estimators': 200}
'''

##########################

'''
#Train model và lưu lại dưới dạng file rf.joblib
rf_model_best = rf(n_estimators=200, max_depth=20, random_state=42)
rf_model_best.fit(X_train, y_train)
dump(rf_model_best, 'rf.joblib')
'''

rf_model_best = load('rf.joblib')


#Đánh giá training set (RF)
y_pred = rf_model_best.predict(X_train)
print("Evaluate training sets (RF): ")
print('R2 score:', r2_score(y_train, y_pred))
print('Pearson R:', pearsonr(y_train, y_pred))
print('RMSE:', np.sqrt(mse(y_train, y_pred)))
print('MAE:', mae(y_train, y_pred))

print('\n')

#Đánh giá training set (IMERG)
print("Đánh giá training set (IMERG): ")
print('Person R:', pearsonr(y_train, z_train))
print('RMSE:', np.sqrt(mse(z_train, y_train)))
print('MAE:', mae(z_train, y_train))

print('\n')

#Đánh giá validation set (RF)
y1_pred = rf_model_best.predict(X_val)
print("Đánh giá validation set (RF): ")
print('R2 score:', r2_score(y_val, y1_pred))
print('Person R:', pearsonr(y_val, y1_pred))
print('RMSE:', np.sqrt(mse(y1_pred, y_val)))
print('MAE:', mae(y1_pred, y_val))

print('\n')

#Đánh giá validation set (IMERG)
print("Đánh giá validation set (IMERGE): ")
print('Person R:', pearsonr(z_val, y_val))
print('RMSE:', np.sqrt(mse(z_val, y_val)))
print('MAE:', mae(z_val, y_val))

print('\n')

#Đánh giá testing set (RF)
y2_pred = rf_model_best.predict(X_test)
print("Đánh giá testing set (RF): ")
print('R2 score:', r2_score(y_test, y2_pred))
print('Person R:', pearsonr(y_test, y2_pred))
print('RMSE:', np.sqrt(mse(y2_pred, y_test)))
print('MAE:', mae(y2_pred, y_test))

print('\n')

#Đánh giá testing set (IMERG)
print("Đánh giá testing set (IMERG): ")
print('Person R:', pearsonr(y_test, z_test))
print('RMSE:', np.sqrt(mse(y_test, z_test)))
print('MAE:', mae(y_test, z_test))


