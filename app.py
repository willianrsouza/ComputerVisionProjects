from flask import Flask, render_template, request, jsonify
import base64
from PIL import Image
import io
import shutil
from src.new_model.modeltest import ImageSearch

app = Flask(__name__)


@app.route("/")
def index():
    return render_template('index.html')


@app.route("/upload", methods=['POST'])
def upload():
    data_url = request.json['croppedImageData']
    image_data = base64.b64decode(data_url.split(',')[1])

    image = Image.open(io.BytesIO(image_data)).convert('RGB')
    image.save('./src/search/img.jpg', 'JPEG')

    update_images()
    return jsonify({'status': 'success', 'message': 'Imagem cortada recebida e salva com sucesso.'})


@app.route('/get_image_paths', methods=['GET'])
def get_image_paths():
    image_paths = update_images()
    return jsonify(image_paths=image_paths)


def update_images():
    model = ImageSearch()
    paths = model.search()

    if paths:
        while len(paths) < 2:
            paths.append(paths[0])

        for i in range(2):
            new_image_path = f'static/image-{i + 1}.jpg'
            shutil.copy(paths[i], new_image_path)

        return [f'static/image-{i+1}.jpg' for i in range(2)]


if __name__ == "__main__":
    app.run()
