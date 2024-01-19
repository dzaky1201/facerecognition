from flask import Flask
from extension import socketio


def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'qwerty123'
    register_extensions(app)
    return app


def register_extensions(app):
    socketio.init_app(app)
