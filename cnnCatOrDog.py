import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import streamlit as st

# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
#import tensorflow as tf
#import tensorflow.keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

st.sidebar.header('Upload Image to Classify')
st.write("""
# Cat or Dog Prediction
""")
uploaded_file_buffer = st.sidebar.file_uploader(label="")
# st.sidebar.button('Upload')
if uploaded_file_buffer is not None:
    uploaded_file = Image.open(uploaded_file_buffer)
    print(uploaded_file.mode)
    uploaded_file = uploaded_file.convert('RGB')
    print(uploaded_file.mode)
    print(type(uploaded_file_buffer))
    print(type(uploaded_file))
    cnn_model = load_model('cat_or_dog.h5')
    st.write('This is the uploaded image:')
    st.image(uploaded_file)
    test_image = uploaded_file.resize((64, 64))
    print(test_image.size)
    test_image = image.img_to_array(test_image)
    print(test_image.shape)
    test_image = test_image/255
    test_image = np.expand_dims(test_image, axis=0)
    print(test_image.shape)
    predicted_class = cnn_model.predict_classes(test_image)[0][0]
    if predicted_class == 1:
        st.write('Predicted Animal: **Dog**')
        st.write('Confidence of Prediction: ', np.round(cnn_model.predict(test_image)[0][0], 2) * 100, '%')
    elif predicted_class == 0:
        st.write('Predicted Animal: **Cat**')
        st.write('Confidence of Prediction: ', np.round((1-cnn_model.predict(test_image)[0][0]), 2) * 100, '%')
    print('Predicted Category: ', predicted_class)
    print('Prediction Probability: ', cnn_model.predict(test_image))
else:
    st.write('After you upload the image, it will appear below')





# print(type(cnn_model.history.history))

