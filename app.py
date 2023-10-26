from flask import Flask, render_template, request, redirect, url_for, jsonify
from dlr_diffusion_tools import text_to_image, image_to_image
from azure_data_writer import upload_file_to_blob
from dotenv import load_dotenv
import os
from io import BytesIO
from config import Supported_Models
load_dotenv()


app = Flask(__name__)

MODEL_ID = os.getenv("MODEL_ID")

@app.route('/', methods=['GET'])
def index():
    return "You've reached the DLR Diffusion API"

@app.route('/generate/text_to_image', methods=['POST'])
def generate_text_to_image():
    # Get the text from the POST request
    model_id = request.json['model_id']
    text = request.json['text']
    num_inference_steps = request.json['num_inference_steps']
    height = request.json['height']
    width = request.json['width']

    # check if the post request isnt using promthero/openjourney model return error
    if model_id not in Supported_Models:
        resp = {
            "success": False,
            "code": 400,
            "message": "Model not supported"
        }
        return jsonify(resp)
    # Process the text and generate the image (dummy code for illustration)
    # Replace this with your actual image generation logic
    image_url = text_to_image(model_id, text, num_inference_steps=num_inference_steps, height=height, width=width)

    # return successful api response with image url
    data = {
        "success": True,
        "code": 200,
        "image_url": image_url
    }
    return jsonify(data)

@app.route('/generate/image_to_image', methods=['POST'])
def generate_image_to_image():
    print(request.form)
    # get the image from the POST request
    model_id = request.form['model_id']
    image_file = request.files['image']
    text = request.form['text']
    negative_prompt = request.form['negative_prompt']
    num_inference_steps = int(request.form['num_inference_steps'])
    strength = float(request.form['strength'])
    guidance_scale = float(request.form['guidance_scale'])

    # check if the post request isnt using promthero/openjourney model return error
    if model_id != "prompthero/openjourney":
        resp = {
            "success": False,
            "code": 400,
            "message": "Model not supported"
        }
        return jsonify(resp)




    # Read the file data and pass it to a BytesIO object
    image_data = image_file.read()
    bytes_io = BytesIO(image_data)

    image_url = image_to_image(model_id,
                               bytes_io,
                               text,
                               negative_prompt,
                               num_inference_steps,
                               strength,
                               guidance_scale
                              )


    resp = {
        "success": True,
        "code": 200,
        "image_url": image_url

    }

    return resp

# creat controlnet endpoint
@app.route('/generate/controlnet', methods=['POST'])
def generate_controlnet():
    # get the image from the POST request
    model_id = request.form['model_id']
    controlnet_type = request.form['controlnet_type']
    image_file = request.files['image']
    text = request.form['text']
    negative_prompt = request.form['negative_prompt']
    num_inference_steps = int(request.form['num_inference_steps'])
    strength = float(request.form['strength'])
    guidance_scale = float(request.form['guidance_scale'])




if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
