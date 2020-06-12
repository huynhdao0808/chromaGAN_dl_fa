import tensorflow as tf
import os

from tensorflow.keras.models import Model
from tensorflow.python.keras.layers import Add, BatchNormalization, Conv2D, Dense, Flatten, Input, LeakyReLU, PReLU, Lambda, MaxPool2D
from tensorflow.keras.optimizers.schedules import PiecewiseConstantDecay
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy, MeanAbsoluteError, MeanSquaredError
from tensorflow.keras.metrics import Mean

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

class ESR(object):
    def __init__(self):
        self.n_filters = 64
        self.inc_filter = 32
        self.channel = 3
        self.n_RRDB_blocks = 15
        self.residual_scaling = 0.2
        self.init_kernel = tf.initializers.he_normal(seed=None)
    
    def pixel_shuffle(self, scale):
        return lambda x: tf.nn.depth_to_space(x, scale)
    
    def upsample(self, x_in, name_up):
        x = Conv2D(self.n_filters, kernel_size=3, padding='same',
                   kernel_initializer=self.init_kernel)(x_in)
        x = Lambda(self.pixel_shuffle(scale=2))(x)
        x = LeakyReLU(0.2)(x)
        
        return x
    
    def conv_RRDB(self, x_in, n_filters, name_rrdb = None, name_dense = None, name_conv=None, activate=True):
        x = Conv2D(n_filters, kernel_size=3, strides=1, padding='same', 
                   kernel_initializer=self.init_kernel)(x_in)
        if activate:
            x = LeakyReLU(0.2)(x)
        return x 
    
    def dense_block(self, x, name_rrdb=None, name_dense=None):
        x1 = self.conv_RRDB(x, self.inc_filter)
        x2 = self.conv_RRDB(tf.concat([x, x1], axis=3), self.inc_filter)
        x3 = self.conv_RRDB(tf.concat([x, x1, x2], axis=3), self.inc_filter)
        x4 = self.conv_RRDB(tf.concat([x, x1, x2, x3], axis=3), self.inc_filter)
        x5 = self.conv_RRDB(tf.concat([x, x1, x2, x3, x4], axis=3), self.n_filters, activate=False)
        
        return x5 * self.residual_scaling
    
    def RRDB(self, x_in, name_rrdb=None):
        """Residual in Residual Dense Block"""
        x_branch = tf.identity(x_in)      
        
        x_branch += self.dense_block(x_branch)
        x_branch += self.dense_block(x_branch)
        x_branch += self.dense_block(x_branch)
        
        x = Add()([x_in, x_branch * self.residual_scaling])
        
        return x 
    
    def build(self, input_shape):
        x_in = Input(shape=input_shape)
        
        x = Conv2D(self.n_filters, kernel_size=3, 
                   strides=1, padding='same',
                   kernel_initializer=self.init_kernel)(x_in)
        
        x_branch = tf.identity(x)
        
        for i in range(self.n_RRDB_blocks):
            x_branch = self.RRDB(x_branch, i)
        
        x_branch = Conv2D(self.n_filters, kernel_size=3, strides=1, padding='same',
                          kernel_initializer=self.init_kernel)(x_branch) 
                            
        x += x_branch
        
        x = self.upsample(x, 1)
        x = self.upsample(x, 2)
        
        x = Conv2D(self.n_filters, kernel_size=3, strides=1, padding='same', 
                   kernel_initializer=self.init_kernel)(x) 
        x = LeakyReLU(0.2, name='LeakyReLU')(x)
        x = Conv2D(self.channel, kernel_size=3, strides=1, padding='same',
                   kernel_initializer=self.init_kernel, activation='sigmoid')(x)
        
        
        return Model(x_in, x)

esr_model = ESR().build(input_shape=(None, None, 3))
print('Create model!')
esr_model.load_weights(os.path.join(MODEL_DIR,'gen_weight.h5'))
print('Loaded weight!')
esr_model.save(os.path.join(MODEL_DIR,'esr.h5'))
print('Saved model!')