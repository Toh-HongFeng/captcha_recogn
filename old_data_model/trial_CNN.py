from tensorflow import keras # type: ignore
from tensorflow.keras import layers, models, regularizers # type: ignore
import os
import scipy.io as sio
import h5py     # version 3.6.0 is last compatible with hdf5, and thus tensorflow
import cv2
import numpy as np

# Load the dataset
#
# citation:
# Cohen, Gregory; Afshar, Saeed; Tapson, Jonathan; van Schaik, Andre (2017): EMNIST: an extension of MNIST to handwritten letters. Western Sydney University. https://doi.org/10.26183/m9k1-zr06
ds = sio.loadmat(os.path.join('emnist-matlab', 'matlab', 'emnist-balanced.mat'))
validate_data_len = ds['dataset']['test'][0, 0]['images'][0, 0].shape[0]
train_images = ds['dataset']['train'][0, 0]['images'][0, 0][:validate_data_len - 1].reshape(validate_data_len - 1, 28, 28, 1)
train_labels = ds['dataset']['train'][0, 0]['labels'][0, 0][:validate_data_len - 1]
val_images = ds['dataset']['train'][0, 0]['images'][0, 0][validate_data_len:].reshape(-1, 28, 28, 1)
val_labels = ds['dataset']['train'][0, 0]['labels'][0, 0][validate_data_len:]

# model = models.Sequential()
# model.add(layers.Conv2D(28, (3, 3), activation='relu', input_shape=(28, 28, 3,)))
# model.add(layers.AveragePooling2D((2, 2)))
# model.add(layers.Conv2D(56, (3, 3), activation='relu'))
# model.add(layers.AveragePooling2D((2, 2)))
# model.add(layers.Conv2D(56, (3, 3), activation='relu'))
# model.add(layers.Flatten())
# model.add(layers.Dropout(0.5))
# model.add(layers.Dense(56, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
# model.add(layers.Dense(47, activation='softmax'))  # Assuming 26 letters (A-Z) + 10 digits (0-9)

# model.compile(optimizer='adam',
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

data_labels = []
for (_, _, files) in os.walk('/content/gdrive/MyDrive/Colab Notebooks/letter_data/letter_data'):
    for file in files:
        data_labels.append(ord(str(file.split('-')[0])))
train_ds = keras.preprocessing.image_dataset_from_directory(
    '/content/gdrive/MyDrive/Colab Notebooks/letter_data/letter_data',
    labels=data_labels,
    validation_split=0.2,
    subset="training",
    seed=110,
    image_size=(28, 28),
    batch_size=1000,
)
val_ds = keras.preprocessing.image_dataset_from_directory(
    '/content/gdrive/MyDrive/Colab Notebooks/letter_data/letter_data',
    labels=data_labels,
    validation_split=0.2,
    subset="validation",
    seed=110,
    image_size=(28, 28),
    batch_size=1000,
)
train_ds = train_ds.prefetch(buffer_size=32)
val_ds = val_ds.prefetch(buffer_size=32)

def SimpleVGG(input_shape):
    inputs = keras.Input(shape=input_shape)

    # entry block
    x = layers.Rescaling(1.0 / 255)(inputs)
    x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    # set aside residual
    previous_block_activation = x

    for size in [128, 256, 512, 728]:

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.Activation("relu")(x)
        x = layers.SeparableConv2D(size, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)

        x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

        # project residual
        residual = layers.Conv2D(size, 1, strides=2, padding="same")(
            previous_block_activation
        )

        # add back residual
        x = layers.add([x, residual])

        # set aside next residual
        previous_block_activation = x

    x = layers.SeparableConv2D(1024, 3, padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)

    x = layers.GlobalAveragePooling2D()(x)
    activation = "sigmoid"
    units = 47

    x = layers.Dropout(0.5)(x)
    x = layers.Dense(25, activation='relu')(x)
    outputs = layers.Dense(units, activation=activation)(x)

    return keras.Model(inputs, outputs)

model = SimpleVGG((28, 28, 1,))
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_ds, epochs=20, validation_data=val_ds)

choice = input('Save the model? [y/n]: ')
if choice.lower() == 'y':
    model.save('emnist_cnn.keras')