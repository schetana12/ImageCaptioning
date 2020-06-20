import streamlit as st 
from PIL import Image
from generator import *
from tensorflow.keras.models import load_model

st.markdown("<h1 style='text-align: center; color: red;'>Image Captioning</h1>", unsafe_allow_html=True)

st.markdown("<h3 style='text-align: center; color: Black;'>Select the model architecture to generate caption</h1>", unsafe_allow_html=True)

model1 = st.button("Merge Architecture")
model2 = st.button("Inject Architecture")
if model1:
	image = Image.open('./merge_architecture.png')
	model = 'Merge'
	st.markdown("<h3 style='text-align: center; color: Black;'>Using {} Architecture to generate captions</h1>".format(model), unsafe_allow_html=True)
	st.image(image, caption='Merge Architecture', use_column_width=True)
if model2:
	image = Image.open('./inject_architecture.png')
	model = 'Inject'
	st.markdown("<h3 style='text-align: center; color: Black;'>Using {} Architecture to generate captions</h1>".format(model), unsafe_allow_html=True)
	st.image(image, caption='Inject Architecture', use_column_width=True)

@st.cache()
def get_model():
	weights = '/home/hs/Desktop/Projects/ImageCaptioning/weights/without_glove/model_40.h5'
	model = load_model(weights)
	return model


uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
	image = Image.open(uploaded_file)
	st.image(image, caption='Uploaded Image.', use_column_width=True)
	st.write("")
	st.write("Generating...")

	#image = load_img(test_sample, target_size=(224, 224))
	model = get_model()
	caption = generate(model, uploaded_file)
	st.write(caption)