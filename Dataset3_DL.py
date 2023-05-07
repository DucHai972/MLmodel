import itertools
import os
from keras import Model, Input
from keras.applications.densenet import layers
from keras.layers import Dropout, Conv2DTranspose, Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from numpy import concatenate
from opt_einsum.backends import tensorflow
from osgeo import gdal
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import load_model
from tensorflow import concat
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import cv2
import matplotlib.pyplot as plt

dataset_path = "./Dataset3"
image_types = ["himawari", "radar", "cape", "isor", "tcc", "tcw", "tcwv"]

data = []
targets = []
count = 0

# Loop through all image types
for image_type in image_types:
    image_type_path = os.path.join(dataset_path, image_type)
    if (count >= 100):
        break
    # Loop through all years
    for year in os.listdir(image_type_path):
        if (count >= 100):
            break
        year_path = os.path.join(image_type_path, year)

        # Loop through all months
        for month in os.listdir(year_path):
            if (count >= 100):
                break
            month_path = os.path.join(year_path, month)

            # Loop through all days
            for day in os.listdir(month_path):
                if (count >= 100):
                    break
                day_path = os.path.join(month_path, day)

                # Loop through all hours
                for hour in os.listdir(day_path):
                    if (count >= 150):
                        break
                    hour_path = os.path.join(day_path, hour)

                    # Loop through all image files in hour folder
                    for filename in os.listdir(hour_path):
                        if (count >= 150):
                            break
                        if filename.endswith(".tif"):
                            # Load image data from file
                            filepath = os.path.join(hour_path, filename)
                            dataset = gdal.Open(filepath)
                            data.append(dataset.ReadAsArray())


                            # Load ảnh radar tương ứng để làm target
                            radar_path = os.path.join(dataset_path, "radar", year, month, day, hour, f"radar_{year}{month}{day}_{hour}.tif")
                            radar_dataset = gdal.Open(radar_path)
                            targets.append(radar_dataset.ReadAsArray())

                            count += 1


# Chuyển data và target sang numpy array
data = np.array(data, dtype=np.float16)
data = np.expand_dims(data, axis=-1)
targets = np.array(targets, dtype=np.float16)
targets = np.expand_dims(targets, axis=-1)

# Chia data -> train, test
train_data, test_data, train_targets, test_targets = train_test_split(data, targets, test_size=0.1, random_state=42)

# Chia data -> train, val
train_data, val_data, train_targets, val_targets = train_test_split(train_data, train_targets, test_size=0.11, random_state=42)

###Visual data
# Plot histogram of pixel values in input data
plt.hist(data.flatten(), bins=50)
plt.title("Pixel Value Histogram")
plt.xlabel("Pixel Value")
plt.ylabel("Frequency")
plt.show()

image_types = ["himawari", "radar", "cape", "isor", "tcc", "tcw", "tcwv"]
num_images = [0, 0, 0, 0, 0, 0, 0]

for i, image_type in enumerate(image_types):
    image_type_path = os.path.join(dataset_path, image_type)
    for year in os.listdir(image_type_path):
        year_path = os.path.join(image_type_path, year)
        for month in os.listdir(year_path):
            month_path = os.path.join(year_path, month)
            for day in os.listdir(month_path):
                day_path = os.path.join(month_path, day)
                for hour in os.listdir(day_path):
                    hour_path = os.path.join(day_path, hour)
                    for filename in os.listdir(hour_path):
                        if filename.endswith(".tif"):
                            num_images[i] += 1

plt.bar(image_types, num_images)
plt.title("Number of Images for Each Type")
plt.xlabel("Image Type")
plt.ylabel("Number of Images")
plt.show()
#############################################

##### U-Net model architecture
inputs = Input(shape=(512, 512, 1))
conv1 = Conv2D(64, 3, activation='relu', padding='same')(inputs)
conv1 = Conv2D(64, 3, activation='relu', padding='same')(conv1)
pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
conv2 = Conv2D(128, 3, activation='relu', padding='same')(pool1)
conv2 = Conv2D(128, 3, activation='relu', padding='same')(conv2)
pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
conv3 = Conv2D(256, 3, activation='relu', padding='same')(pool2)
conv3 = Conv2D(256, 3, activation='relu', padding='same')(conv3)
pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
conv4 = Conv2D(512, 3, activation='relu', padding='same')(pool3)
conv4 = Conv2D(512, 3, activation='relu', padding='same')(conv4)
drop4 = Dropout(0.5)(conv4)
pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
conv5 = Conv2D(1024, 3, activation='relu', padding='same')(pool4)
conv5 = Conv2D(1024, 3, activation='relu', padding='same')(conv5)
drop5 = Dropout(0.5)(conv5)

up6 = Conv2DTranspose(512, 2, strides=(2, 2), padding='same')(drop5)


merge6 = concat([drop4, up6], axis=3)


conv6 = Conv2D(512, 3, activation='relu', padding='same')(merge6)
conv6 = Conv2D(512, 3, activation='relu', padding='same')(conv6)

up7 = Conv2DTranspose(256, 2, strides=(2, 2), padding='same')(conv6)
merge7 = concat([conv3, up7], axis=3)
conv7 = Conv2D(256, 3, activation='relu', padding='same')(merge7)
conv7 = Conv2D(256, 3, activation='relu', padding='same')(conv7)

up8 = Conv2DTranspose(128, 2, strides=(2, 2), padding='same')(conv7)
merge8 = concat([conv2, up8], axis=3)
conv8 = Conv2D(128, 3, activation='relu', padding='same')(merge8)
conv8 = Conv2D(128, 3, activation='relu', padding='same')(conv8)

up9 = Conv2DTranspose(64, 2, strides=(2, 2), padding='same')(conv8)
merge9 = concat([conv1, up9], axis=3)
conv9 = Conv2D(64, 3, activation='relu', padding='same')(merge9)
conv9 = Conv2D(64, 3, activation='relu', padding='same')(conv9)
conv9 = Conv2D(2, 3, activation='relu', padding='same')(conv9)

outputs = Conv2D(1, 1, activation='sigmoid')(conv9)

model = Model(inputs=inputs, outputs=outputs)
model.summary()

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

test_datagen = ImageDataGenerator(rescale=1./255)


train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
)
val_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest',
)

train_generator = train_datagen.flow(
    train_data,
    train_targets,
    batch_size=8,
    shuffle=True
)

validation_generator = val_datagen.flow(
    val_data,
    val_targets,
    batch_size=8,
    shuffle=True
)


test_generator = test_datagen.flow(
        test_data,
        test_targets,
        batch_size= 8,
        shuffle=True
)

fig, ax = plt.subplots(nrows=2, ncols=4, figsize=(12, 6))

#Hiển thị 1 số hình ngẫu nhiên từ training
for i in range(2):
    for j in range(4):
        augmented_image, _ = train_generator.next()
        ax[i, j].imshow(augmented_image[0])
        ax[i, j].axis('off')

plt.show()

##Save model
'''
history = model.fit(
    train_generator,
    steps_per_epoch= 2000,
    epochs=50,
    validation_data= validation_generator,
    validation_steps=800)

model.save('Dataset3_model.h5')
'''

model = load_model('Dataset3_model.h5')
del data
del targets

score = model.evaluate_generator(test_generator, steps=1000)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Lấy vài ảnh từ test_generator
x_test, y_test = test_generator.next()
n_samples = 8  # số lượng ảnh mẫu cần lấy
x_sample, y_sample = x_test[:n_samples], y_test[:n_samples]

# Dự đoán kết quả cho ảnh mẫu bằng model đã huấn luyện
y_pred = model.predict(x_sample)

# Vẽ ảnh mẫu kèm theo nhãn dự đoán và nhãn thực tế
fig, axes = plt.subplots(2, n_samples//2, figsize=(12, 6))
for i, ax in enumerate(axes.flat):
    ax.imshow(x_sample[i])
    ax.set_xlabel(f"Predicted: {np.argmax(y_pred[i])}")
    ax.set_title(f"True: {np.argmax(y_sample[i])}")
    ax.axis('off')
plt.tight_layout()
plt.show()


### Resize images
'''
test_data_resized = []
for image in test_data[:-10]:
    image_resized = cv2.resize(np.uint8(image), (512, 512), interpolation=cv2.INTER_AREA)
    test_data_resized.append(image_resized)

test_targets_resized = []
for target in test_targets[:-10]:
    target_resized = cv2.resize(np.uint8(target), (512, 512), interpolation=cv2.INTER_AREA)
    test_targets_resized.append(target_resized)

print("Resized test_data shape:", np.array(test_data_resized).shape)
print("Resized test_targets shape:", np.array(test_targets_resized).shape)

test_data_resized = np.array(test_data_resized, dtype=test_data.dtype)
test_targets_resized = np.array(test_targets_resized, dtype=test_targets.dtype)
'''


#test_data = test_data[:-10]
#test_targets = test_targets[:-10]

print("Test_data shape:", np.array(val_data).shape)
print("Test_targets shape:", np.array(val_targets).shape)

#Here

print("predicting...")
# predict on validation set
y_pred = model.predict(val_data)


print("Flattening...")
# flatten arrays
y_true = val_targets.flatten().astype('float16')
y_pred = y_pred.flatten().astype('float16')

print("Checking for NaN")
# check for NaN or infinity values in y_true and y_pred
is_finite = np.isfinite(y_true) & np.isfinite(y_pred)

print("Remove NaN")
# remove NaN or infinity values
y_true = y_true[is_finite]
y_pred = y_pred[is_finite]
print("Y_true: ", len(y_true), " and y_pred: ", len(y_pred))


print("calculating...")
# Check if y_true and y_pred have at least one element
if len(y_true) == 0 or len(y_pred) == 0:
    print("Error: y_true or y_pred is empty.")
else:
    # calculate Pearson r
    pearson_r = np.corrcoef(y_true, y_pred)[0, 1]
    print("done person_r")
    # calculate RMSE
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print("done rmse")
    # calculate MAE
    mae = mean_absolute_error(y_true, y_pred)
    print("done mae")

    # print evaluation metrics
    print(f"Pearson r: {pearson_r:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
