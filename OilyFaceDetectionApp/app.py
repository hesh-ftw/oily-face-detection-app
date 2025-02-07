from flask import Flask, render_template, request, url_for, send_from_directory
from tensorflow.keras.preprocessing import image
import numpy as np
import os
import tensorflow as tf

app = Flask(__name__)

# Load the trained model
model_path = "oily_model_final8.h5"  # Update with your model path
model = tf.keras.models.load_model(model_path)

# Define a function for making predictions
def predict_oiliness(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize the image

    # Make predictions
    predictions = model.predict(img_array)

    # Interpret predictions
    class_names = {0: 'non-oily', 1: 'oily'}
    predicted_class = np.argmax(predictions)
    predicted_label = class_names[predicted_class]
    confidence_level = predictions[0][predicted_class] * 100  # Confidence as percentage

    return predicted_label, confidence_level


@app.route('/test-images')
def test_images():
    image_dir = os.path.join(app.static_folder, 'images')
    files = os.listdir(image_dir)
    return {'images_found': files}


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle image upload
        uploaded_file = request.files['file']
        if uploaded_file.filename != '':
            image_path = os.path.join('static', 'uploads', uploaded_file.filename)
            uploaded_file.save(image_path)


            # Make predictions
            predicted_label, confidence_level = predict_oiliness(image_path)

            # Pass prediction results and image path to index.html
            image_url = url_for('static', filename=f'uploads/{uploaded_file.filename}')
            return {
                'image_url': image_url,
                'predicted_label': predicted_label,
                'confidence_level': confidence_level
            }

    return render_template('index.html')




if __name__ == '__main__':
    app.run(debug=True)
