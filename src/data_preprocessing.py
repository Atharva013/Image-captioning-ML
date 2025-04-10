import string
import os

def load_captions(filepath):
    """
    Load the captions from Flickr8k.token.txt and organize into a dictionary.
    """
    captions_dict = {}

    with open(filepath, 'r') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue

            img_id, caption = line.split('\t')
            img_id = img_id.split('#')[0]  # Remove #0, #1, etc.

            # Clean caption
            cleaned_caption = clean_caption(caption)

            # Add start and end tokens
            final_caption = f"startseq {cleaned_caption} endseq"

            # Add to dict
            if img_id not in captions_dict:
                captions_dict[img_id] = []
            captions_dict[img_id].append(final_caption)

    return captions_dict

def clean_caption(caption):
    """
    Lowercase, remove punctuation, and filter short/non-alpha words.
    """
    caption = caption.lower()
    caption = caption.translate(str.maketrans('', '', string.punctuation))
    caption = caption.split()
    caption = [word for word in caption if len(word) > 1 and word.isalpha()]
    return ' '.join(caption)

# For testing from terminal
if __name__ == "__main__":
    captions_file = os.path.join("data", "Flickr8k_text", "Flickr8k.token.txt")
    captions = load_captions(captions_file)

    print(f"Loaded captions for {len(captions)} images.")
    print("\nExample:")
    for img, caps in list(captions.items())[:1]:
        print(f"{img}:")
        for c in caps:
            print(f"  - {c}")
