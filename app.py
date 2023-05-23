import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the trained shape classifier model
model = tf.keras.models.load_model('shape_classifier.h5')

# Dictionary mapping class indices to shape labels
class_labels = {0: 'circle', 1: 'square', 2: 'triangle'}

# Function to preprocess the input image
def preprocess_image(img):
    img = img.resize((64, 64))  # Resize the image to match the input size of the model
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0  # Normalize the pixel values between 0 and 1
    return img

# Streamlit app
def main():
    st.title('Shape Classification App')
    st.write('Upload an image and the app will classify the shape.')

    uploaded_file = st.file_uploader('Choose an image file', type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        img = image.load_img(uploaded_file)
        st.image(img, caption='Uploaded Image', use_column_width=True)

        # Preprocess the image
        processed_img = preprocess_image(img)

        # Make predictions
        prediction = model.predict(processed_img)
        predicted_class = np.argmax(prediction)
        predicted_label = class_labels[predicted_class]

        st.success(f'Predicted Shape: {predicted_label}')

if __name__ == '__main__':
    main()
