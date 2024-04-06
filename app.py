from flask import Flask, render_template, request
from fastai.vision.all import *
from fastai.vision import *
from fastai import *
from PIL import Image 
import sys
import pathlib
from pycaret.regression import *
import pandas as pd
import base64
import io

if sys.platform == "win32":
  temp = pathlib.PosixPath
  pathlib.PosixPath = pathlib.WindowsPath
  del temp


app = Flask(__name__)

vision_model = load_learner('./model/homer_bart_classifier.pkl')
insurance_model = load_model('./model/insurance_gbr_model') 


# Route for the insurance form page (GET request)
@app.route('/', methods=['GET'])
def home():
    return render_template('insurance.html')

# Route for the image classify page (GET request)
@app.route('/image_classify', methods=['GET'])
def bart():
    return render_template('image.html')

# Route for form submission (POST request)
@app.route('/insurance_predict', methods=['POST'])
def process_form():
    data = {}
    data['age'] = [request.form.get('age')]
    data['sex'] = [request.form.get('sex')]
    data['bmi'] = [request.form.get('bmi')]
    data['children'] = [request.form.get('children')]
    data['smoker'] = [request.form.get('smoker')]
    data['region'] = [request.form.get('region')]
    data = pd.DataFrame(data)
    result = predict_model(insurance_model, data=data)  
    return render_template('result.html', result=result.values[0][6])

# Route for uploading an image (POST request)
@app.route('/classify', methods=['POST'])
def upload_image():
    img = Image.open(request.files['input_image'].stream)
    tensor_img = tensor(img)  # converts the image to tensor.
    result = vision_model.predict(tensor_img)[0].capitalize()

    data = io.BytesIO()
    img.save(data, "JPEG")
    encoded_img_data = base64.b64encode(data.getvalue())

    return render_template('result.html', result=result, img_data = encoded_img_data.decode('utf-8'))

if __name__ == '__main__':
    app.run(debug=True)
