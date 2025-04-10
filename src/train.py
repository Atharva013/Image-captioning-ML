import os
import pickle
import numpy as np
import tensorflow as tf

from tensorflow.keras.callbacks import ModelCheckpoint
from data_preprocessing import load_captions
from caption_processing import build_tokenizer, get_max_caption_length
from data_generator import data_generator
from model import build_model

# --- Load everything ---
print("🔹 Loading captions...")
captions = load_captions("data/Flickr8k_text/Flickr8k.token.txt")

print("🔹 Loading image features...")
with open("data/features.pkl", "rb") as f:
    features = pickle.load(f)

print("🔹 Building tokenizer...")
tokenizer = build_tokenizer(captions)
with open("data/tokenizer.pkl", "wb") as f:
    pickle.dump(tokenizer, f)

vocab_size = len(tokenizer.word_index) + 1
max_length = get_max_caption_length(captions)

print(f"✅ Vocab size: {vocab_size}, Max caption length: {max_length}")

# --- Build the model ---
print("🔧 Building model...")
model = build_model(vocab_size, max_length)
model.summary()

# --- Define dataset using tf.data.Dataset ---
print("🌀 Creating TensorFlow dataset...")

batch_size = 32
steps = len(captions) // batch_size

train_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(captions, features, tokenizer, max_length, vocab_size),
    output_signature=(
        (
            tf.TensorSpec(shape=(2048,), dtype=tf.float32),
            tf.TensorSpec(shape=(max_length,), dtype=tf.int32)
        ),
        tf.TensorSpec(shape=(vocab_size,), dtype=tf.float32)
    )
).batch(batch_size).prefetch(tf.data.AUTOTUNE)

# --- Setup checkpoints ---
checkpoint_dir = "checkpoints"
os.makedirs(checkpoint_dir, exist_ok=True)

checkpoint = ModelCheckpoint(
    os.path.join(checkpoint_dir, "model.keras"),
    monitor='loss',
    save_best_only=True,
    verbose=1
)


# --- Train the model ---
print("🚀 Starting training...")
model.fit(
    train_dataset,
    epochs=20,
    steps_per_epoch=steps,
    callbacks=[checkpoint]
)

print("🎉 Training complete. Model saved!")
