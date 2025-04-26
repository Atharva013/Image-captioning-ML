import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # ADD THIS LINE FIRST

from flask import Flask, render_template, request
from utils.helpers import extract_features, generate_caption

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Create upload directory if not exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route("/", methods=["GET", "POST"])
def index():
    caption = None
    image_url = None

    if request.method == "POST":
        try:
            if 'image' not in request.files:
                return "No image uploaded", 400

            image = request.files['image']
            if image.filename == '':
                return "No selected file", 400

            filepath = os.path.join(app.config['UPLOAD_FOLDER'], image.filename)
            image.save(filepath)

            features = extract_features(filepath)
            caption = generate_caption(features)
            image_url = filepath

        except Exception as e:
            print(f"Error: {e}")
            return f"Error processing the image: {str(e)}", 500

    return render_template("index.html", caption=caption, image_url=image_url)

if __name__ == "__main__":
    app.run(debug=True)
