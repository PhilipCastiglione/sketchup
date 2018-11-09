# Sketchup

TODO

# Setup

- macOS
- python 3.6.7

### TensorFlow Research Models

tensorflow setup
proto stuff
modifying file for issue https://github.com/tensorflow/models/issues/4780

### TensorFlow Object Detection

obj detection setup

cocoapi
in models/research

git clone git@github.com:cocodataset/cocoapi.git
cd cocoapi/PythonAPI
pipenv install Cython numpy pycocotools
pipenv run make
cp -r pycocotools ../..

now, pycocotools should be in models/research/

### Sketchup

sketchup needs to be cloned into in models/research/sketchup ??
git submodule something something? or download as zip?


### macOS matplotlib virtualenv workaround

terrible matplotlib virtualenv macosx things

pipenv run python
import matplotlib
matplotlib.matplotlib_fname()
#=> <path to matplotlibrc>

vi <path to matplotlibrc>

comment out backend macosx
add backend: TkAgg

# Usage

### Visualise data

`pipenv run python visualise_bounds.py 5`

### Training

`./train.sh`

