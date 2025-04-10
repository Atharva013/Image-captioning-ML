import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

def data_generator(captions, features, tokenizer, max_length, vocab_size):
    while True:
        for img_caption_key, caps in captions.items():
            img_name = img_caption_key.split('#')[0]  # Fix: remove '.1', '.2', etc.

            if img_name not in features:
                continue  # Skip if features are missing for this image

            img_feat = features[img_name]

            for cap in caps:
                seq = tokenizer.texts_to_sequences([cap])[0]
                for i in range(1, len(seq)):
                    in_seq = seq[:i]
                    out_seq = seq[i]

                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]

                    yield (img_feat, in_seq), out_seq
