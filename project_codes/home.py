from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

# Load the trained model
model_path = 'C:\Users\PC\Documents\Documents\project_codes_agnes'  # Update this path to your model location
model = load_model(model_path)

# Allowed extensions for file upload
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Check if the file has an allowed extension
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Preprocess the image and predict the class
def predict_image(filepath):
    # Load and preprocess the image
    img = load_img(filepath, target_size=(256, 256))  # Ensure this matches your model's expected input size
    img_array = img_to_array(img) / 255.0  # Normalize to [0, 1]
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict using the loaded model
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]

    # Class labels matching the model's output categories
    class_names = ['Mild_Demented', 'Moderate_Demented', 'Non_Brain', 'Non_Demented', 'Very_Mild_Demented']

    return class_names[predicted_class]

# Home route with dementia info and file upload
@app.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        
        # If the user does not select a file, the browser also submits an empty part without a filename
        if file.filename == '':
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join('uploads', filename)  # Ensure 'uploads' directory exists
            file.save(filepath)
            
            # Predict the image class
            prediction = predict_image(filepath)
            
            # Remove the file after prediction to save space
            os.remove(filepath)
            
            return render_template('home.html', prediction=prediction)
    
    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)
