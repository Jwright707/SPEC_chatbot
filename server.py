from flask import Flask, request, jsonify, redirect
from flask_cors import CORS


def create_app():
    app = Flask(__name__)
    return app


# def setup_routes(app):

app = create_app()


@app.route("/", methods=['GET'])
def get():
    return jsonify("Hi")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, threaded=True, debug=False)
