import numpy as np
import pandas as pd
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam
from keras.saving.saving_api import load_model
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae, r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasRegressor

def create_model(learning_rate=0.001):
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1))
    optimizer = Adam(lr=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    return model

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
#Tìm tham số tối ưu và dump mô hình
model = KerasRegressor(build_fn=create_model, verbose=0)

# Define parameter grid
learning_rate = [0.001, 0.01, 0.1]
batch_size = [16, 32, 64]
epochs = [50, 100, 150]
param_grid = dict(learning_rate=learning_rate, batch_size=batch_size, epochs=epochs)

grid = GridSearchCV(estimator = KerasRegressor(build_fn=create_model, epochs=100, batch_size=10, verbose=0, learning_rate=0.001), param_grid=param_grid, n_jobs=-1, cv=3)
grid_result = grid.fit(X_train_val, y_train_val)

# save the best model
best_model = grid_result.best_estimator_.model
best_model.save('NN.h5')
'''


model = load_model('NN.h5')

print("Predicting... ")
y_pred = model.predict(X_train)

print("Đánh giá training set (NN): ")
print('R2 score:', r2_score(y_train, y_pred))
print('RMSE:', np.sqrt(mse(y_pred, y_train)))
print('MAE:', mae(y_pred, y_train))

print('\n')

print("Đánh giá training set (IMERG): ")
print('Person R:', pearsonr(y_train, z_train))
print('RMSE:', np.sqrt(mse(z_train, y_train)))
print('MAE:', mae(z_train, y_train))

print('\n')

y1_pred = model.predict(X_val)

print("Đánh giá validation set (NN): ")
print('R2 score:', r2_score(y_val, y1_pred))
print('RMSE:', np.sqrt(mse(y1_pred, y_val)))
print('MAE:', mae(y1_pred, y_val))

print('\n')

print("Đánh giá validation set (IMERG): ")
print('Person R:', pearsonr(y_val, z_val))
print('RMSE:', np.sqrt(mse(z_val, y_val)))
print('MAE:', mae(z_val, y_val))

print('\n')

y2_pred = model.predict(X_test)
print("Đánh giá testing set (NN): ")
print('R2 score:', r2_score(y_test, y2_pred))
print('RMSE:', np.sqrt(mse(y2_pred, y_test)))
print('MAE:', mae(y2_pred, y_test))

print('\n')

print("Đánh giá testing set (IMERG): ")
print('Person R:', pearsonr(y_test, z_test))
print('RMSE:', np.sqrt(mse(z_test, y_test)))
print('MAE:', mae(z_test, y_test))
