import os
from pathlib import Path
import tensorflow as tf
import numpy as np
import cv2
# from google.colab.patches import cv2_imshow
import datetime
from functools import partial
import re

import tensorflow.keras as keras
from tensorflow.keras import applications
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras import backend as K
from tensorflow.keras.models import load_model, model_from_json, Model
from tensorflow.python.keras.layers.merge import _Merge
from tensorflow.keras.layers import Layer

AUTOTUNE = tf.data.experimental.AUTOTUNE
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
# DIRECTORY INFORMATION
ROOT_DIR = os.path.abspath('.')
MODEL_DIR = os.path.join(ROOT_DIR, 'model')
COL_MODEL_PATH_OTHER = os.path.join(MODEL_DIR, 'my_model_colorization_other.h5')
COL_MODEL_PATH_PEOPLE = os.path.join(MODEL_DIR, 'my_model_colorization_face.h5')
ESR_MODEL_PATH = os.path.join(MODEL_DIR, 'generator_python36.h5')

UPLOAD_DIR = os.path.join(ROOT_DIR,'static','images','upload') # UPDATE
OUT_DIR = os.path.join(UPLOAD_DIR, '..','result')

# DATA INFORMATION
IMAGE_SIZE = 224

def colorization_model():
        img_shape_1 = (IMAGE_SIZE, IMAGE_SIZE, 1)
        img_shape_2 = (IMAGE_SIZE, IMAGE_SIZE, 2)
        img_shape_3 = (IMAGE_SIZE, IMAGE_SIZE, 3)

        img_L_3 = Input(shape= img_shape_3)
        img_L = Input(shape= img_shape_1)
        img_ab_real = Input(shape= img_shape_2)

        input_img = Input(shape=img_shape_3)


        # VGG16 without top layers (Yellow part in Figure 2)
        VGG_model = applications.vgg16.VGG16(weights='imagenet', include_top=False, input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
        model_ = Model(VGG_model.input,VGG_model.layers[-6].output)
        model = model_(input_img)


        # Global Features (Red and Gray parts in Figure 2)

        global_features = keras.layers.Conv2D(512, (3, 3), padding='same', strides=(2, 2), activation='relu')(model)
        global_features = keras.layers.BatchNormalization()(global_features)
        global_features = keras.layers.Conv2D(512, (3, 3), padding='same', strides=(1, 1), activation='relu')(global_features)
        global_features = keras.layers.BatchNormalization()(global_features)

        global_features = keras.layers.Conv2D(512, (3, 3), padding='same', strides=(2, 2), activation='relu')(global_features)
        global_features = keras.layers.BatchNormalization()(global_features)
        global_features = keras.layers.Conv2D(512, (3, 3), padding='same', strides=(1, 1), activation='relu')(global_features)
        global_features = keras.layers.BatchNormalization()(global_features)

        global_features2 = keras.layers.Flatten()(global_features)
        global_features2 = keras.layers.Dense(1024)(global_features2)
        global_features2 = keras.layers.Dense(512)(global_features2)
        global_features2 = keras.layers.Dense(256)(global_features2)
        global_features2 = keras.layers.RepeatVector(28*28)(global_features2)
        global_features2 = keras.layers.Reshape((28,28, 256))(global_features2)
        # For image size 448
#         global_features2 = keras.layers.RepeatVector(56*56)(global_features2)
#         global_features2 = keras.layers.Reshape((56,56, 256))(global_features2)


        global_featuresClass = keras.layers.Flatten()(global_features)
        global_featuresClass = keras.layers.Dense(4096)(global_featuresClass)
        global_featuresClass = keras.layers.Dense(4096)(global_featuresClass)
        global_featuresClass = keras.layers.Dense(1000, activation='softmax')(global_featuresClass)

        # Midlevel Features (The purple part in Figure 2)

        midlevel_features = keras.layers.Conv2D(512, (3, 3),  padding='same', strides=(1, 1), activation='relu')(model)
        midlevel_features = keras.layers.BatchNormalization()(midlevel_features)
        midlevel_features = keras.layers.Conv2D(256, (3, 3),  padding='same', strides=(1, 1), activation='relu')(midlevel_features)
        midlevel_features = keras.layers.BatchNormalization()(midlevel_features)

        # fusion of (VGG16 + Midlevel) + (VGG16 + Global) (Concatenation between Purple and Ped in figure 2)
        modelFusion = keras.layers.concatenate([midlevel_features, global_features2])

        # Fusion + Colorization (The Blue part in Figure 2)
        outputModel =  keras.layers.Conv2D(256, (1, 1), padding='same', strides=(1, 1), activation='relu')(modelFusion)
        outputModel =  keras.layers.Conv2D(128, (3, 3), padding='same', strides=(1, 1), activation='relu')(outputModel)

        outputModel =  keras.layers.UpSampling2D(size=(2,2))(outputModel)
        outputModel =  keras.layers.Conv2D(64, (3, 3), padding='same', strides=(1, 1), activation='relu')(outputModel)
        outputModel =  keras.layers.Conv2D(64, (3, 3), padding='same', strides=(1, 1), activation='relu')(outputModel)

        outputModel =  keras.layers.UpSampling2D(size=(2,2))(outputModel)
        outputModel =  keras.layers.Conv2D(32, (3, 3), padding='same', strides=(1, 1), activation='relu')(outputModel)
        outputModel =  keras.layers.Conv2D(2, (3, 3), padding='same', strides=(1, 1), activation='sigmoid')(outputModel)
        outputModel =  keras.layers.UpSampling2D(size=(2,2))(outputModel)
        final_model = Model(inputs=input_img, outputs = [outputModel, global_featuresClass])

        return final_model

model = colorization_model()
model.load_weights(COL_MODEL_PATH_OTHER)
model.save(os.path.join(MODEL_DIR, 'col_other.h5'))