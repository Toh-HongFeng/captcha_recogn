from tensorflow import keras
from tensorflow.keras import layers, models, regularizers # type: ignore
import os
import h5py     # version 3.6.0 is last compatible with hdf5, and thus tensorflow
import cv2
import numpy as np

# instance segmentation, mask RCNN: https://arxiv.org/pdf/1703.06870
# Backbone: ResNet
def ResNetLayer(x, kernel_size, stride=1):
    y = layers.BatchNormalization()(x)
    y = layers.Activation('relu')(y)
    y = layers.SeparableConv2D(kernel_size*4, 1, strides=stride, padding="same")(x)

    y = layers.BatchNormalization()(y)
    y = layers.Activation('relu')(y)
    y = layers.SeparableConv2D(kernel_size*4, 3, padding="same")(y)

    y = layers.BatchNormalization()(y)
    y = layers.Activation('relu')(y)
    y = layers.SeparableConv2D(kernel_size, 1, padding="same")(y)

    if stride == 1:
        y = layers.add([y, x])
    return y
def ResNet50(input_size):
    input = layers.Input(shape=(input_size[0], input_size[1], 3))
    x = layers.BatchNormalization()(input)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, 7, strides=2, padding="same")(input)
    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

    prev_size = 64
    for kernel_size in [64, 64, 64, 128, 128, 128, 128, 256, 256, 256, 256, 256, 256, 512, 512, 512]:
        stride = 1 if kernel_size == prev_size else 2
        x = ResNetLayer(x, kernel_size, stride)
    return keras.Model(inputs=input, outputs=x)
# Region Proposal Network (RPN)
def RPN(x, num_anchors = 15):
    y = layers.Conv2D(512, (3, 3), padding='same', activation='relu')(x)
    rpn_cls = layers.Conv2D(num_anchors, (1, 1), activation='sigmoid')(y)
    rpn_reg = layers.Conv2D(num_anchors * 4, (1, 1))(y)
    return rpn_reg
# Region of Interest (RoI) Align
def RoIAlign(img, rects):
    for [x, y, w, h] in rects:
        roi = img[y:y+h, x:x+w]
        yield cv2.resize(roi, (28, 28))
# Network Head
def Head(x):
    y = layers.Activation('relu')(x)
    y = layers.SeparableConv2D(1024, 3, padding="same")(y)
    y = layers.BatchNormalization()(y)

    y = layers.Activation('relu')(y)
    y = layers.SeparableConv2D(2048, 3, padding="same")(y)
    y = layers.BatchNormalization()(y)
    
    ave = layers.GlobalAveragePooling2D()(y)
    y = layers.Conv2DTranspose(256, 2, strides=2, padding="same")(y)
    y = layers.BatchNormalization()(y)
    y = layers.SeparableConv2D(80, 1, padding="same")(y)
    mask = layers.BatchNormalization()(y)

    img_class = layers.Dense(2048, activation='softmax')(ave)
    return img_class, mask
# Main
def MaskRCNN():
    input = layers.Input(shape=(80, 1000, 3))
    backbone = ResNet50([80, 1000])

    c1 = backbone(input)
    # c2 = RoIAlign(c1, SelectiveSearch(c1))
    # c2 = RPN(c1)
    c3 = Head(c1)
    return keras.Model(inputs=input, outputs=c3)
def Inceptv3():
    m = keras.applications.InceptionV3(weights='imagenet', input_shape=(80, 1000, 3,))
    x = m.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(2048, activation='relu')(x)
    x = layers.Dense(2048, activation='relu')(x)
    x = layers.Dense(80, activation='softmax')(x)
    return keras.Model(inputs=m.input, outputs=x)


model = Inceptv3()
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# 8010 PNGs in ./train_data
train_images = []
train_labels = []
val_images = []
val_labels = []
for file in os.listdir('./train_data'):
    img = cv2.imread('./train_data/' + file, cv2.IMREAD_COLOR)
    ext_img = np.ndarray((80, 1000, 3), dtype=img.dtype)
    ext_img[:, :img.shape[1]] = img
    ext_img[:, img.shape[1]:] = 255 * np.ones((80, 1000 - img.shape[1], 3), dtype=img.dtype)
    if len(val_images) < 1602:
        val_images.append(ext_img)
        val_labels.append(str(file.split('-')[0]))
    else:
        train_images.append(ext_img)
        train_labels.append(str(file.split('-')[0]))
train_images = np.array(train_images)
train_labels = np.array(train_labels)
val_images = np.array(val_images)
val_labels = np.array(val_labels)

model.fit(train_images, train_labels, epochs=20, validation_data=(val_images, val_labels))

choice = input('Save the model? [y/n]: ')
if choice.lower() == 'y':
    model.save('emnist_cnn.keras')