from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)
try:
    model = load_model('blood_group_model.h5')  
except Exception as e:
    print(f"Error loading model: {e}")
    exit()

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif','bmp'}  
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(128, 128)) 
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img / 255.0
        return img
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])  
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('predict.html', error='No file uploaded') 

        file = request.files['file']
        if file.filename == '':
            return render_template('predict.html', error='No selected file') 

        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)

                img = preprocess_image(file_path)
                if img is None: 
                    return render_template('predict.html', error='Error processing image')

                prediction = model.predict(img)
                blood_groups = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']
                predicted_class = blood_groups[np.argmax(prediction)]

                return render_template('result.html', filename=filename, prediction=predicted_class)

            except Exception as e:
                print(f"Prediction error: {e}")
                return render_template('predict.html', error='An error occurred during prediction') 

        return render_template('predict.html', error='Invalid file format') 
    return render_template('predict.html') 


@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)
