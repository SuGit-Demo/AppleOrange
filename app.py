import os 
from flask import Flask, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

from keras.models import load_model 
from keras.preprocessing import image ###Added
from keras.backend import set_session
from skimage.transform import resize 
import matplotlib.pyplot as plt 
import tensorflow as tf 
import numpy as np 

print("Loading model") 
#global sess
#sess = tf.compat.v1.Session()
#set_session(sess)
global model 
model = load_model('AppleOrangeRGBWorking.h5') 
#global graph
#graph = tf.compat.v1.get_default_graph()

@app.route('/', methods=['GET', 'POST']) 
def main_page():
    if request.method == 'POST':
        file = request.files['file']
        filename = secure_filename(file.filename)
        file.save(os.path.join('uploads', filename))
        return redirect(url_for('prediction', filename=filename))
    return render_template('index.html')

@app.route('/prediction/<filename>') 
def prediction(filename):
    #Step 1
    my_image = plt.imread(os.path.join('uploads', filename))
    #Step 2
    my_image_re = resize(my_image, (32,32,3))
    ########
    img = image.img_to_array(my_image_re)
    img = np.expand_dims(img, axis=0)
    img = img/255
    ########
    
    #Step 3
    #with graph.as_default():
      #set_session(sess)
      #Add
    model.run_eagerly=True  
    probabilities = model.predict(np.array( [my_image_re,] ))[0,:]
    print(probabilities)
    ################
    img_class = model.predict_classes(img) #returns ndim np_array
    img_class_index = img_class.item() #extracting value(s)
    classname = class_names[img_class_index]

    img_prob = model.predict_proba(img) #returns numpy array of class probabilities
    prediction_prob = img_prob.max()

    pred_dict = {"Class":classname, "Probability":prediction_prob}
    print(pred_dict)
    ################
    
    #Step 4
    number_to_class = ['apple', 'orange']
    index = np.argsort(probabilities)
    predictions = {
      "class1":number_to_class[index[1]],
      "class2":number_to_class[index[0]],      
      "prob1":probabilities[index[1]],
      "prob2":probabilities[index[0]],      
     }
    
    #Step 5
    return render_template('predict.html', predictions=predictions)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
