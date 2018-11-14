# Sketchup

PLACEHOLDER

# Setup

Sketchup was built using macOS and Python 3.6.7 on top of the TensorFlow
[Object Detection APIs](https://github.com/tensorflow/models/tree/master/research/object_detection).

I have used [pipenv](https://pipenv.readthedocs.io/) for dependency management,
but it isn't strictly required.

### TensorFlow Object Detection

To use Sketchup requires the TensorFlow models research module.

Install instructions are available
[here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md),
but here are the steps I took.

```
git clone git@github.com:tensorflow/models.git
```

Note: at this time this prototype was built, the repo is at
`d7ce21fa4d3b8b204530873ade75637e1313b760`. At this time, a bug exists that
requires a workaround, if using Python 3. In
`models/research/object_detection/model_lib.py` on line 418 the following change
must be made:

```
-          eval_config, category_index.values(), eval_dict)
+          eval_config, list(category_index.values()), eval_dict)
```

Refer to [this issue](https://github.com/tensorflow/models/issues/4780)
if more details are required.

COCO API is required for our evaluation metrics.

```
cd models/research
git clone git@github.com:cocodataset/cocoapi.git
cd cocoapi/PythonAPI
pipenv install Cython numpy pycocotools
pipenv run make
cp -r pycocotools ../..
```

Now, pycocotools should be present: `models/research/pycocotools`.

The Tensorflow Object Detection API uses Protobufs to configure model and
training parameters. To compile the protobuf libs:

```
brew install protobuf
cd models/research
protoc object_detection/protos/*.proto --python_out=.
```

### Sketchup

Sketchup lives in the TensorFlow `models/research` directory so that it
can access object detection utilities.

```
cd models/resarch
git clone git@github.com:PhilipCastiglione/sketchup.git
pipenv install
```

### macOS matplotlib virtualenv workaround

If you are using macOS and pipenv (or virtualenv) then depending on how you have
installed Python, a terrible workaround may be required for `matplotlib`.

You will get this error when `matplotlib` is invoked if so:

```
RuntimeError: Python is not installed as a framework. The Mac OS X backend will not be able to function correctly if Python is not installed as a framework. See the Python documentation for more information on installing Python as a framework on Mac OS X. Please either reinstall Python as a framework, or try one of the other backends. If you are Working with Matplotlib in a virtual enviroment see 'Working with Matplotlib in Virtual environments' in the Matplotlib FAQ
```

To deal with this you can:

```
pipenv run python
import matplotlib
matplotlib.matplotlib_fname()
#=> <path to matplotlibrc>
exit()

vi <path to matplotlibrc>

# comment out backend macosx
# add backend: TkAgg
```

Refer to [this issue](https://github.com/pyenv/pyenv-virtualenv/issues/140)
and [this SO answer](https://stackoverflow.com/questions/49367013/pipenv-install-matplotlib)
if more details are required.

# Usage

PLACEHOLDER

### Training

The first step is to generate augmentations to the original dataset:

```
pipenv run python augment_data.py
```

It is worthwhile visualising the training data:

```
pipenv run python visualise_bounds.py 10
```

From the images and our data manifests, we generate training data
in the format required by TensorFlow Object Detection:

```
pipenv run python convert_data_for_tf
```

We are now in a position to train:

```
./train.sh
```

You may want to observe training using [TensorBoard](https://www.tensorflow.org/guide/summaries_and_tensorboard):

```
pipenv run tensorboard --logdir=models/model
```

### Prediction

PLACEHOLDER

https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/exporting_models.md

# TODO

train the model
validate it
build the second half, which takes the detected objects and -> DOM

Automate the ridiculous setup, at least for macOS, and make it robust.
readme
document pipeline (provide a tool?)
document 149 data items, test, train validation split
credits
FAQ
licence/disclaimers
