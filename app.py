from flask import Flask, render_template, request, jsonify
from transformers import pipeline
from PIL import Image
import io

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/classify', methods=['POST'])
def classify():
    image = request.files['image']
    image = Image.open(io.BytesIO(image.read())).convert("RGB")

    pipe = pipeline("image-classification", model="dima806/medicinal_plants_image_detection")
    result = pipe(image)

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
