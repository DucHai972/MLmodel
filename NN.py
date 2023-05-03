import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.optimizers import Adam
from joblib import dump, load

data9 = pd.read_excel('Dataset1/data_match9.xlsx', engine='openpyxl')
data10 = pd.read_excel('Dataset1/data_match10.xlsx', engine='openpyxl')

data9.dropna(inplace=True)
data10.dropna(inplace=True)
new_data = pd.concat([data9, data10])
new_data = new_data.sort_values(by=['name', 'datetime'])

X = new_data[['B09B','B10B','B12B','B14B','B16B','I2B','IRB','WVB','CAPE','TCC','TCW','TCWV']]
y = new_data['value']

X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=2/9, random_state=42)

model = Sequential()
model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(1))

model.compile(optimizer=Adam(lr=0.001), loss='mse')

history = model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=50, batch_size=32)

dump(model, 'NN.joblib')

y_pred = model.predict(X_test)
mse_nn = mse(y_test, y_pred)
mae_nn = mae(y_test, y_pred)
corr_nn = np.corrcoef(y_test, y_pred.flatten())[0, 1]

print("MSE: {:.2f}".format(mse_nn))
print("MAE: {:.2f}".format(mae_nn))
print("Correlation: {:.2f}".format(corr_nn))
