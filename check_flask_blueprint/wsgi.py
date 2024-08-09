from flask import Flask, Blueprint, send_from_directory
from flask_restx import Api, Resource

core_bp = Blueprint('core', __name__)
todo_bp = Blueprint('todo', __name__)
user_bp = Blueprint('user', __name__)
api_bp = Blueprint('api', __name__)

api = Api(api_bp)
# core = Api(core_bp)
todo = Api(todo_bp)
user = Api(user_bp)


@core_bp.route('/')
@core_bp.route('/index')
@core_bp.route('/index.html')
def index():
    return send_from_directory("templates/pages", 'index.html')


@todo.route('/todo/<int:id>')
class Todo(Resource):
    def get(self, id):
        return {'todo': f'[{id}] Say "Hello, World!"'}


@todo.route('/todo')
class Todos(Resource):
    def get(self):
        return {'todos': [1,2,3,4]}


@user.route('/user/<int:id>')
class User(Resource):
    def get(self, id):
        return {'user': f'[{id}] Say "Hello, World!"'}


@user.route('/user')
class Users(Resource):
    def get(self):
        return {'users': [1,2,3,4]}


# api.add_resource(TodoItem, '/todo/<int:id>')
# api.add_resource(User, '/user/<int:id>')
# todo.add_resource(Todo, '/todo/<int:id>')
# todo.add_resource(Todos, '/todo')

# user.add_resource(User, '/user/<int:id>')
# user.add_resource(Users, '/user')


def create_app():
    app = Flask(__name__)

    app.register_blueprint(core_bp, url_prefix='/')
    app.register_blueprint(api_bp, url_prefix='/')
    app.register_blueprint(todo_bp, url_prefix='/')
    app.register_blueprint(user_bp, url_prefix='/')
    return app


# app.run(port=8080)
