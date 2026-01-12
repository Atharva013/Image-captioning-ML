
---

# Image Captioning Using Deep Learning

*A CNN–RNN Based Approach with Efficient Data Pipelines and Web Inference*

---

## Abstract

Image captioning is a challenging multimodal problem that lies at the intersection of **computer vision** and **natural language processing**. This project presents an **end-to-end deep learning system** that automatically generates human-readable captions for images. The architecture integrates a **Convolutional Neural Network (CNN)** for visual feature extraction with a **Recurrent Neural Network (RNN)** for sequential language generation.

The system is designed with a strong emphasis on **memory efficiency, modularity, and reproducibility**, employing a **custom Keras DataGenerator**, serialized intermediate artifacts, and a **Flask-based inference interface**. The solution is suitable for academic evaluation, research demonstrations, and portfolio presentation.

---

## Key Contributions

* End-to-end multimodal image captioning pipeline
* Custom **memory-efficient DataGenerator** for large datasets
* Modular CNN–RNN architecture (EfficientNet + LSTM/GRU)
* Training–inference separation for reproducibility
* Web-based inference using Flask
* Support for both **Keras native** and **HDF5** model formats

---

## Tech Stack

**Language**

* Python 3.x

**Deep Learning**

* TensorFlow / Keras
* EfficientNetB0 (CNN encoder)
* LSTM / GRU (caption decoder)

**Web Framework**

* Flask

**Data Handling**

* NumPy
* Pickle (model artifacts & tokenizer)

---

## Dataset

* **Flickr8k Dataset**
* Each image is associated with multiple human-annotated captions
* Images are preprocessed and encoded into feature vectors
* Captions are cleaned, tokenized, and padded for sequence modeling

---

## System Architecture

```
Image
  ↓
EfficientNetB0 (CNN Encoder)
  ↓
Feature Vector
  ↓
Embedding Layer
  ↓
LSTM / GRU Decoder
  ↓
Generated Caption
```

---

## Project Structure

```
image-captioning/
│
├── checkpoints/
│   ├── caption_model.keras      # Final trained captioning model
│   ├── model.h5                 # HDF5-compatible model
│   └── model.keras              # Keras native format
│
├── data/
│   ├── Flickr8k_Dataset/         # Image dataset
│   ├── Flickr8k_text/            # Caption annotation files
│   ├── features.pkl              # Pre-extracted image features
│   └── tokenizer.pkl             # Trained caption tokenizer
│
├── src/
│   ├── caption_processing.py     # Caption cleaning and formatting
│   ├── data_generator.py         # Custom Keras DataGenerator
│   ├── data_preprocessing.py     # Dataset preparation pipeline
│   ├── feature_extraction.py     # CNN-based feature extraction
│   ├── model.py                  # Encoder–Decoder model definition
│   └── train.py                  # Model training script
│
├── static/                        # Static assets (future extension)
│
├── templates/
│   └── index.html                # Flask UI
│
├── utils/
│   └── helpers.py                # Utility and helper functions
│
├── app.py                        # Flask inference application
├── requirements.txt              # Project dependencies
└── README.md                     # Documentation
```

---

## Training vs Inference Flow

### Training Flow

1. **Dataset Preparation**

   * Captions are cleaned and tokenized (`caption_processing.py`)
   * Vocabulary and tokenizer are serialized (`tokenizer.pkl`)

2. **Feature Extraction**

   * Images are passed through EfficientNet
   * Extracted features are stored in `features.pkl`

3. **Batch Generation**

   * Custom `DataGenerator` loads features and captions lazily
   * Prevents RAM overflow during training

4. **Model Training**

   * Encoder–Decoder model trained using teacher forcing
   * Model checkpoints saved in `checkpoints/`

```
Dataset → Preprocessing → Feature Extraction → DataGenerator → Model Training → Saved Model
```

---

### Inference Flow

1. User uploads an image via web interface
2. Image features are extracted using the trained CNN
3. Caption is generated token-by-token using the trained decoder
4. Output caption is displayed in the browser

```
User Image → CNN Feature Extraction → Caption Decoder → Generated Caption
```

---

## Running the Project

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Train the Model

```bash
python src/train.py
```

### Run Inference Server

```bash
python app.py
```

Open in browser:

```
http://127.0.0.1:5000/
```

---

## Sample Output

**Input:** Image of a dog playing in a field
**Output:**

> *A dog is running through the grass.*

---

## Challenges Addressed

* Large-scale dataset memory management
* Training stability on limited hardware (Colab/Kaggle)
* Modularizing ML pipelines for clarity and reuse
* Bridging ML models with real-world applications

---

## Future Work

* Attention mechanism for improved caption quality
* BLEU / METEOR evaluation metrics
* Fine-tuning CNN layers
* REST API deployment
* Dockerized production setup

---

## Applications

* Assistive technology for visually impaired users
* Automatic image indexing and tagging
* Multimedia content analysis
* Intelligent surveillance systems

---

## Author

**Atharva Jadhav**
*Email- atharvajadhav333@gmail.com*

---
