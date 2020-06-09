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
COL_MODEL_PATH_OTHER = os.path.join(MODEL_DIR, 'my_model_colorizationEpoch8.h5')
COL_MODEL_PATH_PEOPLE = os.path.join(MODEL_DIR, 'my_model_colorization.h5')
ESR_MODEL_PATH = os.path.join(MODEL_DIR, 'generator_python36.h5')

UPLOAD_DIR = os.path.join(ROOT_DIR,'static','images','upload') # UPDATE
OUT_DIR = os.path.join(UPLOAD_DIR, '..','result')

# DATA INFORMATION
IMAGE_SIZE = 224

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
        if not os.path.exists(OUT_DIR):
            os.makedirs(OUT_DIR)
        save_path = os.path.join(OUT_DIR, filename)
        cv2.imwrite(save_path, result)
        return result

    def read_img(self, filename):
        img = cv2.imread(filename, 3)
        height, width, channels = img.shape
        min_hw = int(min(height,width)/2)
        img = img[int(height/2)-min_hw:int(height/2)+min_hw,int(width/2)-min_hw:int(width/2)+min_hw,:]
        labimg = cv2.cvtColor(cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE)), cv2.COLOR_BGR2Lab)
        return np.reshape(labimg[:,:,0], (IMAGE_SIZE, IMAGE_SIZE, 1))

    def predict(self, filename):
        file_path = os.path.join(UPLOAD_DIR,filename)

        l_1 = self.read_img(file_path)
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
        image_path = Path(OUT_DIR)
        image_path = image_path.glob('*.*')
        image_name = [path.name for path in image_path]
        image_name = [int(name.split('.')[0]) for name in image_name]
        current_index = max(image_name)
    except: 
        current_index = 0
    return current_index

global img_type
global index
index = get_current_index()
# colorization_model_other = Colorization(COL_MODEL_PATH_OTHER)
# colorization_model_people = Colorization(COL_MODEL_PATH_PEOPLE)
esr_model = ESR(ESR_MODEL_PATH)
path = os.path.join(UPLOAD_DIR,'1.jpg') # path to test image 
print('Load model done')
output = esr_model.predict(path)
print(output.shape)
print('Predict done')
save_path = os.path.join(OUT_DIR, 'test.jpg')
cv2.imwrite(save_path, result)

@app.route('/',methods=['GET','POST'])
def home():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            print('No file part')
            return redirect(request.url)
        data = request.form.to_dict()
        global img_type
        img_type = data['img_type']
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            print('No selected file')
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = file.filename 
            global index 
            index += 1 
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], str(index)+".jpg"))
            new_filename = str(index)+".jpg" 
            return redirect(url_for('result',filename = new_filename))   
    return render_template('home.html', current_index=index)

@app.route('/result/<filename>')
def result(filename):
    global img_type
    if img_type == 1:
        colorization_model_people.predict(filename)
    else:
        colorization_model_other.predict(filename)
    # Prediction(None,filename,float(prediction[0][0])).save_into_db()
    return render_template('result.html', new_filename=filename.split('.')[0]+'.jpg', filename=filename) 

@app.route('/howitworks')
def how_it_works():
    return render_template('how_it_works.html') 

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=5050, debug=True)
 