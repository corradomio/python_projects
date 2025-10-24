# https://realpython.com/flask-blueprint/

# from flask import Flask
#
# app = Flask(__name__)
#
# @app.route('/')
# def index():
#     return "This is an example app"

from flask import Flask
from example_blueprint import example_blueprint

app = Flask(__name__)
app.register_blueprint(example_blueprint)

