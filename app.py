from flask import Flask, render_template, redirect, request, flash, url_for
import tensorflow as tf
import os
import numpy as np
import cv2
import time
from pathlib import Path
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}
# DIRECTORY INFORMATION
ROOT_DIR = os.path.abspath('.')
MODEL_DIR = os.path.join(ROOT_DIR, 'model')
COL_MODEL_PATH_COCO = os.path.join(MODEL_DIR, 'col_coco.h5')
COL_MODEL_PATH_COCO_448 = os.path.join(MODEL_DIR, 'col_coco_448.h5')
COL_MODEL_PATH_CELEBA = os.path.join(MODEL_DIR, 'col_celeba.h5')
COL_MODEL_PATH_IMAGENET = os.path.join(MODEL_DIR, 'col_imagenet.h5')
ESR_MODEL_PATH = os.path.join(MODEL_DIR, 'esr.h5')

UPLOAD_DIR = os.path.join(ROOT_DIR,'static','images','upload') # UPDATE
OUT_DIR_1 = os.path.join(UPLOAD_DIR, '..','result_1')
OUT_DIR_2 = os.path.join(UPLOAD_DIR, '..','result_2')

# DATA INFORMATION
IMAGE_SIZE = 448

app = Flask(__name__)
app.config['UPLOAD_FOLDER']=UPLOAD_DIR    

class ESR(object):
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

    def predict(self, filename):
        file_path = os.path.join(OUT_DIR_1,filename)
        lr = self.preprocess(file_path)
        result = self.model.predict(lr)
        result = (result[0]*255).astype('uint8')
        save_path = os.path.join(OUT_DIR_2, filename)
        plt.imsave(save_path, result)
        return result

class Colorization():
    def __init__(self, model_path):
        self.colorization_model = tf.keras.models.load_model(model_path)
        

    def deprocess(self, imgs):
        imgs = imgs * 255
        imgs[imgs > 255] = 255
        imgs[imgs < 0] = 0
        return imgs.astype(np.uint8)


    def reconstruct(self, batchX, predictedY, filename):
        result = np.concatenate((batchX, predictedY), axis=2)
        result = cv2.cvtColor(result, cv2.COLOR_Lab2BGR)
        result = result[:self.current_shape[0],:self.current_shape[1]]
        if not os.path.exists(OUT_DIR_1):
            os.makedirs(OUT_DIR_1)
        save_path = os.path.join(OUT_DIR_1, filename)
        cv2.imwrite(save_path, result)
        return result

    def read_img(self, filename,img_size):
        img = cv2.imread(filename, 3)
        height, width, channels = img.shape
        max_hw = int(max(height,width))
        img = np.pad(img,((0,max_hw-height),(0,max_hw-width),(0,0)),'median')
        labimg = cv2.cvtColor(cv2.resize(img, (img_size, img_size)), cv2.COLOR_BGR2Lab)
        result_shape =  (np.array([height,width])/max_hw*img_size).astype('int64')
        return np.reshape(labimg[:,:,0], (img_size, img_size, 1)), result_shape

    def predict(self, filename,img_size):
        file_path = os.path.join(UPLOAD_DIR,filename)

        l_1, self.current_shape = self.read_img(file_path,img_size)
        l_1 = np.expand_dims(l_1, axis=0)
        l_1 = l_1/255

        predAB, _  = self.colorization_model.predict(np.tile(l_1,[1,1,1,3]))
        result = self.reconstruct(self.deprocess(l_1[0]),self.deprocess(predAB[0]),filename.split('.')[0]+'.jpg')

        return result

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def get_current_index():
    try:
        image_path = Path(UPLOAD_DIR)
        image_path = image_path.glob('*.*')
        image_name = [path.name for path in image_path]
        image_name = [int(name.split('.')[0]) for name in image_name]
        current_index = max(image_name)
    except: 
        current_index = 0
    return current_index

global index
index = get_current_index()
colorization_model = Colorization(COL_MODEL_PATH_COCO_448)
colorization_model_celeba = Colorization(COL_MODEL_PATH_CELEBA)
esr_model = ESR(ESR_MODEL_PATH)

@app.route('/',methods=['GET','POST'])
def home():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            print('No file part')
            return redirect(request.url)
        data = request.form.to_dict()
        img_type = data['img_type']
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            print('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            global index 
            index += 1 
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], str(index)+".jpg"))
            new_filename = str(index)+".jpg" 
            return redirect(url_for('result',filename = new_filename, img_type = img_type))   
    return render_template('home.html', current_index=index)

@app.route('/result/<filename>/<img_type>')
def result(filename, img_type):
    if int(img_type) == 1:
        colorization_model_celeba.predict(filename,224)
    else:
        colorization_model.predict(filename,448)
    esr_model.predict(filename)
    return render_template('result.html', filename=filename) 

@app.route('/howitworks')
def how_it_works():
    return render_template('how_it_works.html') 

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=5050, debug=True)
 