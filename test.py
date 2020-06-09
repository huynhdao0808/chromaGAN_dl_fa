import tensorflow as tf
import numpy as np
import cv2
import os
import time
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
# DIRECTORY INFORMATION
ROOT_DIR = os.path.abspath('.')
MODEL_DIR = os.path.join(ROOT_DIR, 'model')
MODEL_PATH = os.path.join(MODEL_DIR, 'my_model_colorizationEpoch8.h5')
ESR_PATH = os.path.join(MODEL_DIR, 'gen_model_2000(f2k_2202).h5')

UPLOAD_DIR = os.path.join(ROOT_DIR,'static','images','upload') # UPDATE
OUT_DIR = os.path.join(UPLOAD_DIR, '..','result')

# DATA INFORMATION
IMAGE_SIZE = 224
class Predict(object):
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)
    
    def preprocess(self, path):
        """ input: string
              path to test image
            output: tensor
              shape of [1, height, width, 3]
        """
        lr = tf.io.read_file(path)
        lr = tf.image.decode_image(lr, channels=3)
        lr = tf.cast(lr, dtype=tf.float32) / 255 
        lr = tf.expand_dims(lr, axis=0)

        return lr

    def predict(self, path):
        lr = self.preprocess(path)
        return self.model.predict(lr)

def visualize_prediction(sr):
    sr = (sr[0]*255).astype(int)
    plt.figure(figsize=(10, 8))
    plt.imshow(sr)
    plt.show()

def save_prediction(sr, file_name):
    sr = (sr[0]*255).astype(int)
    plt.imsave('{}.png'.format(file_name), sr)

print(ESR_PATH)
# predictor = Predict(ESR_PATH)
model = tf.keras.models.load_model(ESR_PATH)
# output = predictor.predict('/static/images/result/1.jpg')
# print(output.shape)