from flask import Flask, request, render_template, jsonify
import os
from keras.models import load_model
from keras.preprocessing import image
import numpy as np
from PIL import Image

app = Flask(__name__)

UPLOAD_FOLDER = 'predict'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model = load_model('C:/Users/202-5/Documents/my_model.h5')

def predict_skin_disease(image_path):
    img = Image.open(image_path)
    img = img.resize((100, 75))
    img = np.array(img)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    predictions = model.predict(img)
    predicted_class = np.argmax(predictions, axis=1)

    if predicted_class == 0:
        result = '이 결과는 정상 피부를 나타낼 수 있습니다.'
        accuracy = float(predictions[0][predicted_class])
    else:
        result = '이 결과는 피부암의 가능성을 나타낼 수 있습니다.'
        accuracy = float(predictions[0][predicted_class])

    return result, accuracy

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'result': '파일이 없습니다.', 'accuracy': 0.0})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'result': '파일 이름이 없습니다.', 'accuracy': 0.0})
    if file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        result, accuracy = predict_skin_disease(filename)
        return jsonify({'result': result, 'accuracy': accuracy})

if __name__ == '__main__':
    app.run(debug=True, host='   .168.0.11')