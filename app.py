from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import os

app = Flask(__name__)
model = load_model('poultry_model.h5')

# Ensure uploads folder exists
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file uploaded.'

    file = request.files['file']
    if file.filename == '':
        return 'No selected file.'

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    img = image.load_img(filepath, target_size=(150, 150))  # set to match model input
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
     # Flatten it to match Dense layer input
     
    prediction = model.predict(img_array)[0]
    classes = ['Coccidiosis', 'Healthy', 'New Castle Disease', 'Salmonella']
    predicted_label = classes[np.argmax(prediction)]

    return render_template('index.html', prediction=predicted_label)

if __name__ == '__main__':
    app.run(debug=True)
