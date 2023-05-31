from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from models import generateAnxietyResponse, generateAnyModelResponse, generateClassificationPrediction
import os


app = Flask(__name__)
CORS(app)

@app.route('/execute_generative_model/', methods=['POST'])
def execute_generative_model():
    content = request.json
    print("Contenido de la solicitud: {}".format(content))

    prompt = content['prompt']
    model_name = content['modelo_seleccionado']

    # Llamada a la función generativa
    model_response = generateAnyModelResponse(model_name, prompt)

    response = jsonify({"prompt":content['prompt'], "model": model_name, "res":model_response})
    # response.headers.add('Access-Control-Allow-Origin', '*')

    return response

@app.route('/list_available_models/', methods=['POST'])
def list_available_models():
    #content = request.json

    directorio = "models"

    # Obtener la lista de carpetas dentro del directorio
    carpetas = [nombre for nombre in os.listdir(directorio) if os.path.isdir(os.path.join(directorio, nombre))]

    response = {"modelos": carpetas}
    # response.headers.add('Access-Control-Allow-Origin', '*')
    return response


@app.route('/images/', methods=['POST'])
def images():
    #content = request.json


    response = {"TODO": "TODO"}

    return response

@app.route('/execute_classification_model/', methods=['POST'])
def execute_classification_model():
    content = request.json
    print("Contenido de la solicitud: {}".format(content))

    # Llamada a la función
    pred = generateClassificationPrediction(content["prompt"])

    response = jsonify({"prediccion":pred})
    # response.headers.add('Access-Control-Allow-Origin', '*')
    return response

@app.route('/generate_image/', methods=['GET'])
def generate_image():
    content = request.json
    print("Contenido de la solicitud: {}".format(content))

    # Llamada a la función
    path = r"images\\anxiety\\0.png"

    return send_file(path, mimetype='image/png')


if __name__ == '__main__':
    app.run(host= '0.0.0.0',debug=True)