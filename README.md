# DLRDiffusionAPI

# Requirements
python 3.11.6

# Reference
https://github.com/lllyasviel/ControlNet

# models and datasets
https://huggingface.co/lllyasviel/ControlNet/tree/main
https://huggingface.co/webui/ControlNet-modules-safetensors/tree/main

# Commands

conda env create -f environment.yaml
conda activate control

conda env update -f environment.yaml


pip install --upgrade transformers
pip install opencv-python==4.5.5.64

pip uninstall opencv-python-headless

pip uninstall tying-extension
pip install tying-extension==4.8.0

pip install annotator


python -m venv <Virtual Environment Name>
cd ./venv/Scripts/activate
cd..
cd..
pip install -r requirements.txt
pip install flask
## install diffusers module manually
pip install accelerate
pip install git+https://github.com/huggingface/diffusers

flask run
