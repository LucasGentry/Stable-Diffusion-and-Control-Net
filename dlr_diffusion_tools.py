from diffusers import DiffusionPipeline, StableDiffusionImg2ImgPipeline, StableDiffusionControlNetPipeline
from azure_data_writer import account_name, container_name, upload_file_to_blob
from uuid import uuid4
import os
from PIL import Image
import torch
from dotenv import load_dotenv
from config import Supported_Models

load_dotenv()

ENV = os.getenv("ENV")
MOUNT_PATH = os.getenv("MOUNT_PATH")


def text_to_image(model_id, text, num_inference_steps=50, height=512, width=512):
    print(f'Using Model: {model_id}')
    if model_id not in Supported_Models:
        return "This model is not supported for image to image generation"

    if model_id == "prompthero/openjourney":
        print(os.listdir())
        print(f"Loading from Local {model_id}")
        print(f"{MOUNT_PATH}/openjourney")
        print(os.listdir(f"{MOUNT_PATH}/openjourney"))

        pipe = DiffusionPipeline.from_pretrained(f"{MOUNT_PATH}/openjourney", torch_dtype=torch.float16, local_files_only=True)

    elif model_id == "stabilityai/stable-diffusion-xl-base-1.0":
        print(os.listdir())
        print(f"Loading from Local {model_id}")
        print(f"{MOUNT_PATH}/stable-diffusion-xl-base-1.0")
        print(os.listdir(f"{MOUNT_PATH}/stable-diffusion-xl-base-1.0"))

        pipe = DiffusionPipeline.from_pretrained(f"{MOUNT_PATH}/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16)

    pipe.to("cuda")

    image = pipe(text, num_inference_steps=num_inference_steps, height=height, width=width).images[0]
    unique_id = str(uuid4())
    im_name = "t2i_" + text.replace(" ", "_").replace(",", "-") + "_" + unique_id + ".png"
    image.save(f"static/{im_name}")
    print(f"Writting image to blob storage: {im_name}")
    blob_url = upload_file_to_blob(im_name, f"static/{im_name}")
    # delete the file from the local folder
    os.remove(f"static/{im_name}")

    return blob_url

def image_to_image(model_id, input_image_bytes, text, negative_prompt,num_inference_steps=50, strength=0.75, guidance_scale=7.5):
    print(f'Using Model: {model_id}')
    if model_id != "prompthero/openjourney":
        return "This model is not supported for image to image generation"
    input_image = Image.open(input_image_bytes).convert("RGB")
    if model_id == "prompthero/openjourney":
        print(f"Loading from Local {model_id}")
        pipe = StableDiffusionImg2ImgPipeline.from_pretrained("./openjourney", torch_dtype=torch.float16).to(
            'cuda')

    init_image = Image.open(input_image_bytes).convert("RGB")
    generator = torch.Generator(device='cuda').manual_seed(1024)

    image = pipe(prompt=text, image=init_image,
                 negative_prompt=negative_prompt,
                 num_inference_steps=num_inference_steps,
                 strength=0.75,
                 guidance_scale=7.5,
                 generator=generator).images[0]

    unique_id = str(uuid4())
    im_name = "i2i_" + text.replace(" ", "_").replace(",", "-") + "_" + unique_id + ".png"
    image.save(f"static/{im_name}")
    print(f"Writting image to blob storage: {im_name}")
    blob_url = upload_file_to_blob(im_name, f"static/{im_name}")
    # delete the file from the local folder
    os.remove(f"static/{im_name}")

    return blob_url


def controlnet(
        model_id,
        controlnet_type,
        input_image_bytes,
        text,
        negative_prompt,
        num_inference_steps=50,
        strength=0.75,
        guidance_scale=7.6):

    print(f'Using Model: {model_id}')
    print(f'Using Controlnet: {controlnet_type}')

    input_image = Image.open(input_image_bytes).convert("RGB")





if __name__ == "__main__":
    model_id = "runwayml/stable-diffusion-v1-5"
    text = "A bagel in the style of andy warhol"
    image = text_to_image(model_id, text, num_inference_steps=20, height=512, width=768)
    print(image)

