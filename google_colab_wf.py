from tensorflow import keras
from tensorflow.keras import layers, models, regularizers # type: ignore
import os
from google.colab import drive
import scipy.io as sio
import h5py
import cv2
import numpy as np

drive.mount('/content/gdrive')
!ls

train_ds = keras.preprocessing.image_dataset_from_directory(
    '/content/gdrive/MyDrive/Colab Notebooks/letter_data',
    labels='inferred',
    validation_split=0.2,
    subset="training",
    seed=110,
    image_size=(300, 300),
    batch_size=64,
)
val_ds = keras.preprocessing.image_dataset_from_directory(
    '/content/gdrive/MyDrive/Colab Notebooks/letter_data',
    labels='inferred',
    validation_split=0.2,
    subset="validation",
    seed=110,
    image_size=(300, 300),
    batch_size=64,
)
train_ds = train_ds.prefetch(buffer_size=32)
val_ds = val_ds.prefetch(buffer_size=32)

def SSD():
    input = layers.Input(shape=(300, 300, 3))
    classes = 36

    ## Backbone: VGG16
    # Conv1, L1
    C1 = layers.Rescaling(1.0 / 255)(input)
    C1 = layers.Conv2D(32, 3, strides=2, padding="same")(C1)
    C1 = layers.BatchNormalization()(C1)
    C1 = layers.Activation("relu")(C1)
    # L2
    C1 = layers.Conv2D(64, 3, padding="same")(C1)
    C1 = layers.BatchNormalization()(C1)
    C1 = layers.Activation("relu")(C1)

    # Conv2, L1
    C2 = layers.Activation("relu")(C1)
    C2 = layers.SeparableConv2D(128, 3, padding="same")(C2)
    C2 = layers.BatchNormalization()(C2)
    # L2
    C2 = layers.Activation("relu")(C2)
    C2 = layers.SeparableConv2D(128, 3, padding="same")(C2)
    C2 = layers.BatchNormalization()(C2)
    # L3
    C2 = layers.MaxPooling2D(3, strides=2, padding="same")(C2)
    res2 = layers.Conv2D(128, 1, strides=2, padding="same")(C1)
    C2 = layers.add([C2,res2])

    # Conv3, L1
    C3 = layers.Activation("relu")(C2)
    C3 = layers.SeparableConv2D(256, 3, padding="same")(C3)
    C3 = layers.BatchNormalization()(C3)
    # L2
    C3 = layers.Activation("relu")(C3)
    C3 = layers.SeparableConv2D(256, 3, padding="same")(C3)
    C3 = layers.BatchNormalization()(C3)
    # L3
    C3 = layers.MaxPooling2D(3, strides=2, padding="same")(C3)
    res3 = layers.Conv2D(256, 1, strides=2, padding="same")(C2)
    C3 = layers.add([C3,res3])

    # Conv4, L1
    C4 = layers.Activation("relu")(C3)
    C4 = layers.SeparableConv2D(512, 3, padding="same")(C4)
    C4 = layers.BatchNormalization()(C4)
    # L2
    C4 = layers.Activation("relu")(C4)
    C4 = layers.SeparableConv2D(512, 3, padding="same")(C4)
    C4 = layers.BatchNormalization()(C4)
    # L3
    C4 = layers.MaxPooling2D(3, strides=2, padding="same")(C4)
    res4 = layers.Conv2D(512, 1, strides=2, padding="same")(C3)
    C4 = layers.add([C4,res4])
    S1 = layers.Conv2D(4*(classes + 4), 3, padding='same')(C4)

    # Conv5, L1
    C5 = layers.Activation("relu")(C4)
    C5 = layers.SeparableConv2D(728, 3, padding="same")(C5)
    C5 = layers.BatchNormalization()(C5)
    # L2
    C5 = layers.Activation("relu")(C5)
    C5 = layers.SeparableConv2D(728, 3, padding="same")(C5)
    C5 = layers.BatchNormalization()(C5)
    # L3
    C5 = layers.MaxPooling2D(3, strides=2, padding="same")(C5)
    res5 = layers.Conv2D(728, 1, strides=2, padding="same")(C4)
    C5 = layers.add([C5,res5])

    # Conv6
    C6 = layers.Activation("relu")(C5)
    C6 = layers.SeparableConv2D(1024, 3, padding="same")(C6)
    C6 = layers.BatchNormalization()(C6)
    # Conv7
    C7 = layers.Activation("relu")(C6)
    C7 = layers.SeparableConv2D(1024, 1, padding="same")(C7)
    C7 = layers.BatchNormalization()(C7)
    S2 = layers.Conv2D(6*(classes + 4), 3, padding='same')(C7)

    # Conv8, L1
    C8 = layers.Activation("relu")(C7)
    C8 = layers.SeparableConv2D(256, 1, padding="same")(C8)
    C8 = layers.BatchNormalization()(C8)
    # L2
    C8 = layers.MaxPooling2D(3, strides=2, padding="same")(C8)
    C8 = layers.SeparableConv2D(512, 3, padding="same")(C8)
    C8 = layers.BatchNormalization()(C8)
    res8 = layers.Conv2D(512, 3, strides=2, padding="same")(C7)
    C8 = layers.subtract([C8,res8])
    S3 = layers.Conv2D(6*(classes + 4), 3, padding='same')(C8)

    # Conv9, L1
    C9 = layers.Activation("relu")(C8)
    C9 = layers.SeparableConv2D(128, 1, padding="same")(C9)
    C9 = layers.BatchNormalization()(C9)
    # L2
    C9 = layers.MaxPooling2D(3, strides=2, padding="same")(C9)
    C9 = layers.SeparableConv2D(256, 3, padding="same")(C9)
    C9 = layers.BatchNormalization()(C9)
    res9 = layers.Conv2D(256, 3, strides=2, padding="same")(C8)
    C9 = layers.subtract([C9,res9])
    S4 = layers.Conv2D(6*(classes + 4), 3, padding='same')(C9)

    # Conv10, L1
    C10 = layers.Activation("relu")(C9)
    C10 = layers.SeparableConv2D(128, 1, padding="same")(C10)
    C10 = layers.BatchNormalization()(C10)
    # L2
    C10 = layers.MaxPooling2D(3, strides=2, padding="same")(C10)
    C10 = layers.SeparableConv2D(256, 3, padding="same")(C10)
    C10 = layers.BatchNormalization()(C10)
    res10 = layers.Conv2D(256, 3, strides=2, padding="same")(C9)
    C10 = layers.subtract([C10,res10])
    S5 = layers.Conv2D(4*(classes + 4), 3, padding='same')(C10)

    # Conv11, L1
    C11 = layers.Activation("relu")(C10)
    C11 = layers.SeparableConv2D(128, 1, padding="same")(C11)
    C11 = layers.BatchNormalization()(C11)
    # L2
    C11 = layers.MaxPooling2D(3, strides=2, padding="same")(C11)
    C11 = layers.SeparableConv2D(256, 3, padding="same")(C11)
    C11 = layers.BatchNormalization()(C11)
    res11 = layers.Conv2D(256, 3, strides=2, padding="same")(C10)
    C11 = layers.subtract([C11,res11])
    S6 = layers.Conv2D(4*(classes + 4), 3, padding='same')(C11)

    # Detection
    S1 = layers.GlobalAveragePooling2D()(S1)
    S1 = layers.Dropout(0.5)(S1)
    S1 = layers.Dense(512, activation='relu')(S1)

    S2 = layers.GlobalAveragePooling2D()(S2)
    S2 = layers.Dropout(0.5)(S2)
    S2 = layers.Dense(512, activation='relu')(S2)

    S3 = layers.GlobalAveragePooling2D()(S3)
    S3 = layers.Dropout(0.5)(S3)
    S3 = layers.Dense(512, activation='relu')(S3)

    S4 = layers.GlobalAveragePooling2D()(S4)
    S4 = layers.Dropout(0.5)(S4)
    S4 = layers.Dense(512, activation='relu')(S4)

    S5 = layers.GlobalAveragePooling2D()(S5)
    S5 = layers.Dropout(0.5)(S5)
    S5 = layers.Dense(512, activation='relu')(S5)

    S6 = layers.GlobalAveragePooling2D()(S6)
    S6 = layers.Dropout(0.5)(S6)
    S6 = layers.Dense(512, activation='relu')(S6)

    # Non-maximum suppression
    x = layers.Concatenate()([S1,S2,S3,S4,S5,S6])
    x = layers.Dense(classes, activation='sigmoid')(x)
    return keras.Model(inputs=input, outputs=x)

model = SSD()
# model.summary()
model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_ds, epochs=20, validation_data=val_ds)

model.save('/content/gdrive/MyDrive/Colab Notebooks/SSD_letter.keras')

# SSD on letter_data, limit batch_size at 128 due to RAM
# 40 Epoch, batch_size 32: overfit after epoch 3
# 20 Epoch, batch_size 128: overfit after epoch 12
# 20 Epoch, batch_size 100: lr=0.0005, overfit after epoch 9

# SSD2 on letter2_data, batch_size = 64, learning_rate = 0.0001
# Final: 20 Epoch, overfit after epoch 3, train_time = 2h 40m 23s
# Val_acc: 0.6750, Val_loss: 1.9737