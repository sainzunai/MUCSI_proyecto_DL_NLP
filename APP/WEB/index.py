from flask import *
import datetime

app = Flask(__name__, static_url_path='/static')

@app.route('/') # Se llama a la función sobre la URL principal
def home():
    return render_template('index.html')


@app.route('/classification') # Se llama a la función sobre la URL específica de datos
def classification():
    
    return render_template('classificator.html')

@app.route('/generative_images') # Se llama a la función sobre la URL específica de datos
def generative_images():
    
    return render_template('image_visualizer.html')



if __name__ == '__main__':
   app.run(host='0.0.0.0', port=8000, debug=True)