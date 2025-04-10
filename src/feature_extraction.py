import os
import numpy as np
from tqdm import tqdm
import pickle
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing import image

def build_feature_extractor():
    """
    Loads the InceptionV3 model and removes the final classification layer.
    """
    model = InceptionV3(weights='imagenet')
    model_new = Model(inputs=model.input, outputs=model.layers[-2].output)  # 2048-d feature vector
    return model_new

def preprocess_image(img_path):
    """
    Load an image and preprocess it for InceptionV3.
    """
    img = image.load_img(img_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def extract_features_from_folder(img_folder, model):
    """
    Extract features from all images in the given folder using InceptionV3.
    """
    features = {}
    img_names = os.listdir(img_folder)
    
    for img_name in tqdm(img_names, desc="Extracting features"):
        img_path = os.path.join(img_folder, img_name)
        try:
            img_array = preprocess_image(img_path)
            feature = model.predict(img_array, verbose=0)
            features[img_name] = feature.flatten()
        except Exception as e:
            print(f"Error processing {img_name}: {e}")
    
    return features

def save_features(features, save_path="data/features.pkl"):
    with open(save_path, 'wb') as f:
        pickle.dump(features, f)

# Run from terminal
if __name__ == "__main__":
    img_folder = os.path.join("data", "Flickr8k_Dataset")
    print("🔄 Loading InceptionV3 model...")
    model = build_feature_extractor()
    print("✅ Model loaded. Starting feature extraction...")
    
    features = extract_features_from_folder(img_folder, model)
    save_features(features)

    print(f"✅ Features extracted for {len(features)} images and saved to data/features.pkl")
