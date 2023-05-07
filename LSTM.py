import numpy as np
import pandas as pd
from keras import Sequential
from keras.callbacks import ModelCheckpoint
from keras.engine.input_layer import InputLayer
from keras.layers import LSTM, Dense
from keras.losses import mse
from keras.metrics import RootMeanSquaredError, mae
from keras.optimizers import Adam
from keras.utils.vis_utils import plot_model
from keras.models import load_model
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

# Load data
data_9 = pd.read_excel('./Dataset1/data_match9.xlsx', engine='openpyxl')
data_10 = pd.read_excel('./Dataset1/data_match10.xlsx', engine='openpyxl')
data = pd.concat([data_9, data_10], axis=0)

num_cols = [col for col in data.columns if data[col].dtypes != 'O']

# Xử lí các ô giá trị trống
data[num_cols] = data[num_cols].fillna(data[num_cols].median())

# Chuyển format cho datetime
data.index = pd.to_datetime(data['datetime'], format='%Y.%m.%d %H:%M:%S')

# Xo ô giá trị trống
data.drop(columns=['datetime', 'id', 'name', 'lat', 'lon'], inplace=True)

poll = data['value']
poll.plot()

def convert_numpy(df, window_size):
    df_as_np = df.to_numpy()
    X = []
    y = []
    for i in range(len(df_as_np)-window_size):
        row = [[a] for a in df_as_np[i:i+5]]
        X.append(row)
        label = df_as_np[i+5]
        y.append(label)
    return np.array(X), np.array(y)

# Convert X, y
WINDOW_SIZE = 10
X, y = convert_numpy(poll, WINDOW_SIZE)

# Chia data -> training, validating
X_train, y_train = X[:140000], y[:140000]
X_val, y_val = X[140000:], y[140000:650000]
_, X_test, _, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Model architecture
model = Sequential()
model.add(InputLayer((5, 1)))
model.add(LSTM(64))
model.add(Dense(8, activation='relu'))
model.add(Dense(1, activation='linear'))
model.summary()

'''
# Compile model
checkpoint = ModelCheckpoint('my_model/', save_best_only=True)
model.compile(loss='mse', optimizer=Adam(learning_rate=0.0001), metrics=[RootMeanSquaredError()])

# Train model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10, verbose=1, callbacks=[checkpoint])
model.save('LSTM.h5')
'''

model = load_model('LSTM.h5')

train_predictions = model.predict(X_train).flatten()
y_train = np.array(y_train)
train_rmse = np.sqrt(mse(y_train, train_predictions))
print("Đánh giá tập training: ")
print("Training RMSE = ", train_rmse)
print("Training Person R = ", pearsonr(y_train, train_predictions))
print("Training MAE = ", mae(train_predictions, y_train))
print("Training R2 = ", r2_score(train_predictions, y_train))

validation_predictions = model.predict(X_val).flatten()
y_val = np.array(y_val)
print("Đánh giá tập validation: ")
print("Validation RMSE = ", np.sqrt(mse(validation_predictions, y_val)))
print("Validation Person R = ", pearsonr(validation_predictions, y_val))
print("Validation MAE = ", mae(validation_predictions, y_val))
print("Validation R2 = ", r2_score(validation_predictions, y_val))

testing_predictions = model.predict(X_test).flatten()
y_test = np.array(y_test)
print("Đánh giá tập testing: ")
print("Testing RMSE = ", np.sqrt(mse(testing_predictions, y_test)))
print("Testing Person R = ", pearsonr(testing_predictions, y_test))
print("Testing MAE = ", mae(testing_predictions, y_test))
print("Testing R2 = ", r2_score(testing_predictions, y_test))

# lọc ra giá trị tốt nhất cho tập train
train_best_idx = np.where(abs(train_predictions - y_train) <= 200)[0]
train_best_predictions = train_predictions[train_best_idx]
train_best_actual = y_train[train_best_idx]

# vẽ biểu đồ cho tập train
plt.scatter(train_best_actual, train_best_predictions, color='blue', label='Train')
plt.plot([min(train_best_actual), max(train_best_actual)], [min(train_best_actual), max(train_best_actual)], color='black', linestyle='--')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Best Predictions for Train Set')
plt.legend()
plt.show()
