from flask import Flask, render_template, request, redirect, url_for, jsonify
from dlr_diffusion_tools import text_to_image, image_to_image
from azure_data_writer import upload_file_to_blob
from dotenv import load_dotenv
import os
from io import BytesIO
from config import Supported_Models
load_dotenv()

# libs for canny2image, seg2image
from share import *
import config

import cv2
import einops
import gradio as gr
import numpy as np
import torch
import random

from pytorch_lightning import seed_everything
from annotator.util import resize_image, HWC3
from annotator.canny import CannyDetector
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
#


# Preload for canny2image
apply_canny = CannyDetector()

model_canny = create_model('./models/cldm_v15.yaml').cpu()
model_canny.load_state_dict(load_state_dict('./models/control_sd15_canny.pth', location='cuda'))
model_canny = model_canny.cuda()
ddim_sampler_canny = DDIMSampler(model_canny)
#

# Preload for seg2image
apply_uniformer = UniformerDetector()

model_seg = create_model('./models/cldm_v15.yaml').cpu()
model_seg.load_state_dict(load_state_dict('./models/control_sd15_seg.pth', location='cuda'))
model_seg = model_seg.cuda()
ddim_sampler_seg = DDIMSampler(model_seg)
#


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

# create canny2image endpoint
@app.route('/generate/canny_to_image', methods=['POST'])
def generate_canny_to_image():
    input_image = request.form['input_image']
    prompt = request.form['prompt']
    a_prompt = request.form['a_prompt']
    n_prompt = request.form['n_prompt']
    num_samples = request.form['num_samples']
    image_resolution = request.form['image_resolution']
    ddim_steps = request.form['ddim_steps']
    guess_mode = request.form['guess_mode']
    strength = request.form['strength']
    scale = request.form['scale']
    seed = request.form['seed']
    eta = request.form['eta']
    low_threshold = request.form['low_threshold']
    high_threshold = request.form['high_threshold']
    with torch.no_grad():
        img = resize_image(HWC3(input_image), image_resolution)
        H, W, C = img.shape

        detected_map = apply_canny(img, low_threshold, high_threshold)
        detected_map = HWC3(detected_map)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model_canny.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model_canny.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model_canny.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model_canny.low_vram_shift(is_diffusing=True)

        model_canny.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler_canny.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model_canny.low_vram_shift(is_diffusing=False)

        x_samples = model_canny.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    # return successful api response with image url
    data = {
        "success": True,
        "code": 200,
        "image_url": [255 - detected_map] + results
    }
    return jsonify(data)
    
# create seg2image endpoint
@app.route('/generate/seg_to_image', methods=['POST'])
    input_image = request.form['input_image']
    prompt = request.form['prompt']
    a_prompt = request.form['a_prompt']
    n_prompt = request.form['n_prompt']
    num_samples = request.form['num_samples']
    image_resolution = request.form['image_resolution']
    detect_resolution = request.form['detect_resolution']
    ddim_steps = request.form['ddim_steps']
    guess_mode = request.form['guess_mode']
    strength = request.form['strength']
    scale = request.form['scale']
    seed = request.form['seed']
    eta = request.form['eta']

    with torch.no_grad():
        input_image = HWC3(input_image)
        detected_map = apply_uniformer(resize_image(input_image, detect_resolution))
        img = resize_image(input_image, image_resolution)
        H, W, C = img.shape

        detected_map = cv2.resize(detected_map, (W, H), interpolation=cv2.INTER_NEAREST)

        control = torch.from_numpy(detected_map.copy()).float().cuda() / 255.0
        control = torch.stack([control for _ in range(num_samples)], dim=0)
        control = einops.rearrange(control, 'b h w c -> b c h w').clone()

        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        if config.save_memory:
            model_seg.low_vram_shift(is_diffusing=False)

        cond = {"c_concat": [control], "c_crossattn": [model_seg.get_learned_conditioning([prompt + ', ' + a_prompt] * num_samples)]}
        un_cond = {"c_concat": None if guess_mode else [control], "c_crossattn": [model_seg.get_learned_conditioning([n_prompt] * num_samples)]}
        shape = (4, H // 8, W // 8)

        if config.save_memory:
            model_seg.low_vram_shift(is_diffusing=True)

        model_seg.control_scales = [strength * (0.825 ** float(12 - i)) for i in range(13)] if guess_mode else ([strength] * 13)  # Magic number. IDK why. Perhaps because 0.825**12<0.01 but 0.826**12>0.01
        samples, intermediates = ddim_sampler_seg.sample(ddim_steps, num_samples,
                                                     shape, cond, verbose=False, eta=eta,
                                                     unconditional_guidance_scale=scale,
                                                     unconditional_conditioning=un_cond)

        if config.save_memory:
            model_seg.low_vram_shift(is_diffusing=False)

        x_samples = model_seg.decode_first_stage(samples)
        x_samples = (einops.rearrange(x_samples, 'b c h w -> b h w c') * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)

        results = [x_samples[i] for i in range(num_samples)]
    # return successful api response with image url
    data = {
        "success": True,
        "code": 200,
        "image_url": [detected_map] + results
    }
    return jsonify(data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
