#!/bin/bash
pip3 install torch==1.10.0+cu111 torchvision==0.11.0+cu111 torchaudio==0.10.0 -f https://download.pytorch.org/whl/torch_stable.html
pip3 uninstall mmaction2 -y
pip3 uninstall mmcv-full -y

# conda install av -c conda-forge -y
pip3 install av
pip3 install PyTurboJPEG moviepy decord==0.4.1
pip3 install einops timm prettytable nvidia-ml-py3 Cython numpy matplotlib Pillow six terminaltables IPython mmpycocotools albumentations>=0.3.2
pip3 install -U fire lmdb tqdm sklearn pandas
pip3 install --upgrade --force-reinstall opencv-python-headless
pip3 install -U opencv-python-headless


pip3 install mmcv-full==1.3.18 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html


pip3 install -r requirements/build.txt
pip3 install -r requirements/optional.txt
pip3 install -r requirements/runtime.txt
pip3 install -r requirements/tests.txt
pip3 install -v -e .
pip3 install --upgrade --force-reinstall opencv-python-headless

pip3 install transformers==4.6.1
pip3 install gpustat
pip3 install tensorboard
pip3 install nltk
pip3 install ffmpeg-python

pip3 install SceneGraphParser
python3 -m spacy download en