from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load trained model
try:
    model = load_model('blood_group_model.h5')  # Make sure the path is correct
except Exception as e:
    print(f"Error loading model: {e}")
    exit() # Exit if the model can't be loaded

# Define allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif','bmp'}  # Added gif
UPLOAD_FOLDER = 'static/uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path):
    try:
        img = image.load_img(img_path, target_size=(128, 128)) # Consistent target size
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = img / 255.0  # Normalize
        return img
    except Exception as e:
        print(f"Error preprocessing image: {e}")
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])  # Added GET for initial page load
def predict():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('predict.html', error='No file uploaded') # Render with error

        file = request.files['file']
        if file.filename == '':
            return render_template('predict.html', error='No selected file') # Render with error

        if file and allowed_file(file.filename):
            try:
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)

                img = preprocess_image(file_path)
                if img is None: # Handle preprocessing errors
                    return render_template('predict.html', error='Error processing image')

                prediction = model.predict(img)
                blood_groups = ['A+', 'A-', 'B+', 'B-', 'AB+', 'AB-', 'O+', 'O-']
                predicted_class = blood_groups[np.argmax(prediction)]

                return render_template('result.html', filename=filename, prediction=predicted_class)

            except Exception as e:
                print(f"Prediction error: {e}")
                return render_template('predict.html', error='An error occurred during prediction') # More general error

        return render_template('predict.html', error='Invalid file format') # Render with error
    return render_template('predict.html') # For GET request, just show the upload form


@app.route('/about')
def about():
    return render_template('about.html')

if __name__ == '__main__':
    app.run(debug=True)