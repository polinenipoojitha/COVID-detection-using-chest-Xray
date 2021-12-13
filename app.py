

from flask import Flask, render_template, request, session, redirect, url_for, flash
import os
from utilities.get_result import get_result
from werkzeug.utils import secure_filename
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
import cv2
import numpy as np

UPLOAD_FOLDER = './flask app/assets/images'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg', 'gif'])

app = Flask(__name__,static_url_path='/assets',
            static_folder='./flask app/assets', 
            template_folder='./flask app')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/')
def root():
   return render_template('index.html')

@app.route('/index.html')
def index():
   return render_template('index.html')

@app.route('/upload.html')
def upload():
   return render_template('upload.html')


@app.route('/upload_chest.html')
def upload_chest():
   return render_template('upload_chest.html')

@app.route('/uploaded_chest', methods = ['POST', 'GET'])
def uploaded_chest():
   if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        # if user does not select file, browser also
        # submit a empty part without filename
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            # filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], 'upload_chest.jpg'))

   resnet_chest = load_model('models/resnet_chest.h5')
   vgg_chest = load_model('models/vgg_chest.h5')
   inception_chest = load_model('models/inceptionv3_chest.h5')
   xception_chest = load_model('models/xception_chest.h5')
   created_model_large_chest  = load_model('models/created_model_large.h5')
   created_model_medium_chest  = load_model('models/created_model_medium.h5')
   created_model_small_chest  = load_model('models/created_model_small.h5')


   image = cv2.imread('./flask app/assets/images/upload_chest.jpg') # read file 
   image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # arrange format as per keras
   image = cv2.resize(image,(224,224))
   image = np.array(image) / 255
   image = np.expand_dims(image, axis=0)
   
   resnet_pred = resnet_chest.predict(image)
   probability = get_result(resnet_pred[0])
   if probability[0] > 0.5:
      resnet_chest_pred = str('%.2f' % (probability[0]*100) + '% COVID')   
   elif probability[0] > 0.4 and probability[0] < 0.5:
      resnet_chest_pred = str('%.2f' % (probability[0]*100) + '% VIRAL PNEUMONIA')
   else:
      resnet_chest_pred = str('%.2f' % ((1-probability[0])*100) + '% Non COVID')
   print(resnet_chest_pred)

   vgg_pred = vgg_chest.predict(image)
   probability = vgg_pred[0]
   print("VGG Predictions:")
   if probability[0] > 0.5:
      vgg_chest_pred = str('%.2f' % (probability[0]*100) + '% COVID')   
   elif probability[0] > 0.4 and probability[0] < 0.5:
      vgg_chest_pred = str('%.2f' % (probability[0]*100) + '% VIRAL PNEUMONIA')
   else:
      vgg_chest_pred = str('%.2f' % ((1-probability[0])*100) + '% Non COVID')
   print(vgg_chest_pred)

   inception_pred = inception_chest.predict(image)
   probability = inception_pred[0]
   print("Inception Predictions:")
   if probability[0] > 0.5:
      inception_chest_pred = str('%.2f' % (probability[0]*100) + '% COVID')  
   elif probability[0] > 0.4 and probability[0] < 0.5:
      inception_chest_pred = str('%.2f' % (probability[0]*100) + '% VIRAL PNEUMONIA')
   else:
      inception_chest_pred = str('%.2f' % ((1-probability[0])*100) + '% Non COVID')
   print(inception_chest_pred)

   xception_pred = xception_chest.predict(image)
   probability = xception_pred[0]
   print("Xception Predictions:")
   if probability[0] > 0.5:
      xception_chest_pred = str('%.2f' % (probability[0]*100) + '% COVID')   
   elif probability[0] > 0.4 and probability[0] < 0.5:
      xception_chest_pred = str('%.2f' % (probability[0]*100) + '% VIRAL PNEUMONIA')
   else:
      xception_chest_pred = str('%.2f' % ((1-probability[0])*100) + '% Non COVID')
   print(xception_chest_pred)

   created_model_large_pred = created_model_large_chest.predict(image)
   probability = get_result(created_model_large_pred[0])
   if probability[0] > 0.5:
      created_model_chest_large_pred = str('%.2f' % (probability[0]*100) + '% COVID')  
   elif probability[0] > 0.4 and probability[0] < 0.5:
      created_model_chest_large_pred = str('%.2f' % (probability[0]*100) + '% VIRAL PNEUMONIA')
   else:
      created_model_chest_large_pred = str('%.2f' % ((1-probability[0])*100) + '% Non COVID')

   created_model_medium_pred = created_model_medium_chest.predict(image)
   probability = created_model_medium_pred[0]
   if probability[0] > 0.5:
      created_model_chest_medium_pred = str('%.2f' % (probability[0]*100) + '% COVID')  
   elif probability[0] > 0.4 and probability[0] < 0.5:
      created_model_chest_medium_pred = str('%.2f' % (probability[0]*100) + '% VIRAL PNEUMONIA')
   else:
      created_model_chest_medium_pred = str('%.2f' % ((1-probability[0])*100) + '% Non COVID')


   created_model_small_pred = created_model_small_chest.predict(image)
   probability = get_result(created_model_small_pred[0])
   print (probability)
   if probability[0] > 0.5:
      created_model_chest_small_pred = str('%.2f' % (probability[0]*100) + '% COVID') 
   elif probability[0] > 0.4 and probability[0] < 0.5:
      created_model_chest_small_pred = str('%.2f' % (probability[0]*100) + '% VIRAL PNEUMONIA')
   else:
      created_model_chest_small_pred = str('%.2f' % ((1-probability[0])*100) + '% Non COVID')


   return render_template('results_chest.html',resnet_chest_pred=resnet_chest_pred,vgg_chest_pred=vgg_chest_pred,inception_chest_pred=inception_chest_pred,xception_chest_pred=xception_chest_pred,created_model_chest_small_pred= created_model_chest_small_pred, created_model_chest_medium_pred=created_model_chest_medium_pred, created_model_chest_large_pred= created_model_chest_large_pred)

if __name__ == '__main__':
   app.secret_key = ".."
   app.run(port=5051, debug=True)