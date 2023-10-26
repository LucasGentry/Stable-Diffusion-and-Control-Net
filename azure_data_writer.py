from azure.storage.blob import BlobServiceClient, ContentSettings
import os
from dotenv import load_dotenv
load_dotenv()

account_name = os.getenv("ACCOUNT_NAME")
account_key = os.getenv("ACCOUNT_KEY")
container_name = os.getenv("CONTAINER_NAME")

blob_service_client = BlobServiceClient(account_url=f"https://{account_name}.blob.core.windows.net", credential=account_key)

def upload_file_to_blob(filename, filecontent, container_name=container_name):
    with open(filecontent, "rb") as file:
        blob_client = blob_service_client.get_blob_client(container=container_name, blob=filename)
        blob_client.upload_blob(file, content_settings=ContentSettings(content_type='image/png'))

    return blob_client.url
