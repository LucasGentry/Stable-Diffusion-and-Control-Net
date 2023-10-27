# DLRDiffusionAPI

# Requirements
python 3.11.6

# Reference
https://github.com/lllyasviel/ControlNet

# models and datasets
https://huggingface.co/lllyasviel/ControlNet/tree/main
https://huggingface.co/webui/ControlNet-modules-safetensors/tree/main

# Commands
## Using venv

python -m venv <Virtual Environment Name>
cd ./venv/Scripts/activate
cd..
cd..
pip install -r requirements.txt
flask run
## Using Anaconda env
conda env create -f environment.yaml
conda activate control

conda env update -f environment.yaml
