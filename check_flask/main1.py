from flask import Flask
from flask_restx import Resource, Api


class ThisApp(Api):

    def render_root(self):
        return {'hello': 'world'}


app = Flask(__name__)
api = ThisApp(app, doc="/api")


class HelloWorld(Resource):
    def get(self):
        return {'hello': 'world'}


api.add_resource(HelloWorld, '/')

if __name__ == '__main__':
    app.run(debug=True)
