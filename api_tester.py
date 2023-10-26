import requests
import os
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient
from azure_data_writer import blob_service_client
load_dotenv()

API_URL = os.getenv("API_URL")

def query_api(api_url, model_id, text, num_inference_steps=50, height=512, width=512):
    payload = {
        "model_id": model_id,
        "text": text,
        "num_inference_steps": num_inference_steps,
        "height": height,
        "width": width
    }
    response = requests.post(api_url, json=payload)
    print(response)
    return response.json()

if __name__ == "__main__":
    api_url = 'http://127.0.0.1:5000/generate/text_to_image'

    text = "A house in the style of andy warhol"
    # model_id = "runwayml/stable-diffusion-v1-5"
    # model_id = "prompthero/openjourney"
    model_id = "stabilityai/stable-diffusion-2-1"
    resp = query_api(api_url,
                     model_id,
                     text,
                     num_inference_steps=10,
                     height=512,
                     width=512)
    print(resp)
    image_url = resp["image_url"]

    # read the image from the azure blob storage url


    # Specify the container and blob name
    container_name = image_url.split("/")[-2]
    blob_name = image_url.split("/")[-1]

    blob_client = blob_service_client.get_blob_client(container=container_name, blob=blob_name)

    with open(f"MJ_{text}.png", "wb") as file:
        file.write(blob_client.download_blob().readall())

    print("Image downloaded successfully!")