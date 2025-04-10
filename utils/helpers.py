import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image
import pickle

# Load tokenizer and model once
tokenizer = pickle.load(open("data/tokenizer.pkl", "rb"))
model = tf.keras.models.load_model("checkpoints/model.keras")

# Load InceptionV3 model for feature extraction
image_model = InceptionV3(weights='imagenet')
image_model = tf.keras.Model(image_model.input, image_model.layers[-2].output)

# Set max length based on your training
max_length = 34  # Make sure this matches what was used in training

def extract_features(img_path):
    img = Image.open(img_path).resize((299, 299)).convert('RGB')
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    feature = image_model.predict(x, verbose=0)
    return feature

def generate_caption(photo, tokenizer=tokenizer, max_length=max_length):
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = None
        for w, index in tokenizer.word_index.items():
            if index == yhat:
                word = w
                break
        if word is None:
            break
        in_text += ' ' + word
        if word == 'endseq':
            break
    final_caption = in_text.split()[1:-1]  # remove startseq and endseq
    return ' '.join(final_caption)
