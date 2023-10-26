model_id = "runwayml/stable-diffusion-v1-5"
text = "A bagel in the style of andy warhol"
image_url = text_to_image(model_id, text)
print(image_url)
This code will generate an image based on the provided text and print the URL where the generated image is stored.