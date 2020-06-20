import glob
from PIL import Image
import numpy as np
import pickle
from pickle import load
from tqdm import tqdm_notebook
import pandas as pd
import cv2
import os
from os import listdir, path
import pickle
import io
from IPython.display import clear_output, display

#Preprocessing
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

#VGG
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg16 import decode_predictions
from tensorflow.keras.layers import Input

#Layers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.layers import add
from tensorflow.keras.callbacks import ModelCheckpoint


def get_VGG_model():
	# loading VGG model weights
	in_layer = Input(shape=(224, 224, 3))
	model = VGG16(include_top=True, input_tensor=in_layer, weights='imagenet')
	model = Model(model.input, model.layers[-2].output)
	#model.summary()
	return model

def extract_features(model, test_sample=None):

	# loading image

	img = Image.open(test_sample)
	img = img.convert('RGB')
	image = img.resize((224, 224), Image.NEAREST)

	#image = load_img(test_sample, target_size=(224, 224))

	# convert the image pixels to a numpy array
	image = img_to_array(image)
	
	# reshape data for the model
	image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
	
	# prepare the image for the VGG model
	image = preprocess_input(image)
	
	# extracting features using VGG16
	feature = model.predict(image)
	# shape = [1, 7, 7, 512]

	return feature

# define the captioning model
def define_model(vocab_size, max_length):
	# feature extractor model
	#inputs1 = Input(shape=(7,7,512))
	#inp1 = Conv2D(512, (3,3), activation='relu')(inputs1)
	#inp11 = MaxPooling2D(pool_size=(2, 2))(inp1)
	#inp2 = Flatten()(inp11)
	
	# feature extractor model
	#inputs1 = Input(shape=(7,7,512), dtype='float32')
	#inp2 = Flatten(dtype='float32')(inputs1)
	#inp2 = Dense(4096, activation='relu', dtype='float32')(inp2)
	
	# feature extractor model
	inputs1 = Input(shape=(4096,), dtype='float32')
	
	#inputs1 = Input(shape=(2048,))
	fe1 = Dropout(0.5)(inputs1)
	fe2 = Dense(256, activation='relu', dtype='float32')(fe1)
	# sequence model
	inputs2 = Input(shape=(max_length,), dtype='float32')
	se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
	se2 = Dropout(0.5)(se1)
	se3 = LSTM(256, dtype='float32')(se2)
	# decoder model
	decoder1 = add([fe2, se3])
	decoder2 = Dense(256, activation='relu', dtype='float32')(decoder1)
	outputs = Dense(vocab_size, activation='softmax', dtype='float32')(decoder2)
	# tie it together [image, seq] [word]
	model = Model(inputs=[inputs1, inputs2], outputs=outputs)
	# compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam')
	# summarize model
	model.summary()
	#plot_model(model, to_file='model.png', show_shapes=True)
	return model

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def generate_desc(model, tokenizer, photo, max_length):
    # seed the generation process
    in_text = 'startseq'
    # iterate over the whole length of the sequence
    for _ in range(max_length): #why not more than max_length
        # integer encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad input
        sequence = pad_sequences([sequence], maxlen=max_length)
        # predict next word
        yhat = model.predict([photo,sequence], verbose=0)
        # convert probability to integer
        yhat = np.argmax(yhat)
        # map integer to word
        word = word_for_id(yhat, tokenizer)
        # stop if we cannot map the word
        #if word is None: #how is this possible ?
        #    break
        # append as input for generating the next word
        in_text += ' ' + word
        # stop if we predict the end of the sequence
        if word == 'endseq':
            break
    return in_text

# remove start/end sequence tokens from a summary
def cleanup_summary(summary):
    # remove start of sequence token
    index = summary.find('startseq ')
    if index > -1:
        summary = summary[len('startseq '):]
    # remove end of sequence token
    index = summary.find(' endseq')
    if index > -1:
        summary = summary[:index]
    return summary


def generate(model, sample):
	model_VGG=get_VGG_model()
	test_features=extract_features(model_VGG, test_sample=sample)

	# merge model params
	max_length = 34
	# loading tokenizer
	with open('/home/hs/Desktop/Projects/ImageCaptioning/tokenizer.pickle', 'rb') as handle:
		tokenizer = pickle.load(handle)
		vocab_size = len(tokenizer.word_index) + 1

	#model = define_model(vocab_size, max_length)
	#weights = '/home/hs/Desktop/Projects/ImageCaptioning/weights/without_glove/model_40.h5'
	#model = load_model(weights)
	pred = generate_desc(model, tokenizer, test_features, max_length)
	return cleanup_summary(pred)

if __name__ == '__main__':
	image = Image.open('/home/hs/Desktop/Projects/ImageCaptioning/test_samples/2.jpg')
	imgByteArr = io.BytesIO()
	image.save(imgByteArr, format='PNG')
	path = '/home/hs/Desktop/Projects/ImageCaptioning/test_samples/2.jpg'
	print(generate(imgByteArr))