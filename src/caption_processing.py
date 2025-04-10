import os
import pickle
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def build_tokenizer(captions_dict, vocab_size=5000):
    """
    Creates and fits a tokenizer on all captions.
    """
    all_captions = []
    for caption_list in captions_dict.values():
        all_captions.extend(caption_list)

    tokenizer = Tokenizer(num_words=vocab_size, oov_token="<unk>")
    tokenizer.fit_on_texts(all_captions)

    return tokenizer

def get_max_caption_length(captions_dict):
    """
    Find the maximum caption length in the dataset.
    """
    all_captions = []
    for caption_list in captions_dict.values():
        all_captions.extend(caption_list)

    max_len = max(len(caption.split()) for caption in all_captions)
    return max_len

def save_tokenizer(tokenizer, filepath="data/tokenizer.pkl"):
    with open(filepath, 'wb') as f:
        pickle.dump(tokenizer, f)

# Example test run
if __name__ == "__main__":
    from data_preprocessing import load_captions

    captions_file = os.path.join("data", "Flickr8k_text", "Flickr8k.token.txt")
    captions_dict = load_captions(captions_file)

    tokenizer = build_tokenizer(captions_dict)
    save_tokenizer(tokenizer)

    max_len = get_max_caption_length(captions_dict)
    print(f"✅ Tokenizer saved. Max caption length: {max_len}")
