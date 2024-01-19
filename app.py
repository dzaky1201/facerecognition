from flask import Flask, render_template
from flask_socketio import SocketIO, emit
from keras.models import load_model
import cv2
import numpy as np
from config import create_app
from extension import socketio

app = create_app()

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Load the model
model = load_model("./dataset/keras_Model.h5", compile=False)

# Load the labels
class_names = open("./dataset/labels.txt", "r").readlines()

# CAMERA can be 0 or 1 based on the default camera of your computer
camera = cv2.VideoCapture(0)


@app.route('/')
def index():
    return render_template('index.html')


@socketio.on('request_prediction')
def handle_request_prediction():
    global emit_flag
    emit_flag = True
    while emit_flag:
        ret, frame = camera.read()
        image = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
        image_input = np.asarray(
            image, dtype=np.float32).reshape(1, 224, 224, 3)
        image_input = (image_input / 127.5) - 1
        prediction = model.predict(image_input)
        index = np.argmax(prediction)
        class_name = class_names[index]
        confidence_score = prediction[0][index]
        _, img_encoded = cv2.imencode('.jpg', frame)
        img_bytes = img_encoded.tobytes()
        emit('prediction', {'image': img_bytes, 'class_name': class_name[2:], 'confidence': str(
            np.round(confidence_score * 100))[:-2]})
    emit('disconnect', {'image': 'no_video.png'})


@socketio.on('disconnect_request')
def disconnect_request():
    global emit_flag
    emit_flag = False


if __name__ == '__main__':
    app.run()
