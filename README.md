# Sketchup

...

This project was inspired by work done by the Microsoft AI Lab in a project called
[Sketch 2 Code](https://www.ailab.microsoft.com/experiments/30c61484-d081-4072-99d6-e132d362b99d)
([code repository](https://github.com/Microsoft/ailab/tree/master/Sketch2Code))
and an original [paper](https://arxiv.org/abs/1705.07962) and project by Tony
Beltramelli called [pix2code](https://uizard.io/research/#pix2code)
([code repository](https://github.com/tonybeltramelli/pix2code)).

## Dataset

...

The dataset is made available thanks to the Microsoft AI Lab under the MIT License.
The AI Lab repository can be found [here](https://github.com/Microsoft/ailab) and
the data [here](https://github.com/Microsoft/ailab/tree/master/Sketch2Code/model).

## Setup

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

### Transfer Learning

So that training doesn't start form scratch, which takes a very long time for object detection models,
we use a pretrained model from the
[Tensorflow detection model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
as a starting point for our training.

```
curl http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28.tar.gz > models/model/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28.tar.gz
tar -xzvf models/model/faster_rcnn_inception_resnet_v2_atrous_coco_2018_01_28.tar.gz -C models/model/
```

[This example](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/running_pets.md#downloading-a-coco-pretrained-model-for-transfer-learning)
also demonstrates the use of a pretrained model for transfer learning (though using GCP).

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

## Usage

### Training

#### Data Augmentation

First, augment the original image dataset.

```
pipenv run python scripts/augment_dataset.py
```

Note that the current default is to only include the "Button" label, the flag
`--all-labels` can be passed to include all 10 classes. This flag will also
need to be passed to the training script.

This will take the original 149 images in `./data/original_images/` and apply
every combination of six stochastic augmentations. Bounding boxes are also
transformed equivalently.

Augmentations are applied using
[this library](https://github.com/aleju/imgaug) as seen
[here](src/augmenter.py#L16).

A total of 9,536 images will now be in `./data/images/` and an augmented
manifest will be at `./data/dataset.json`.

#### Visualize Augmented Training Data

It is recommended that you visualize labelled training data.

```
pipenv run python scripts/visualize_input_dataset.py 20
```

This will display 20 of the training images randomly sampled from all of them,
with bounding boxes drawn around labeled elements in the image. The bounding
box colors are generated each run.

#### Convert Data For TensorFlow

TensorFlow Object Detection requires a particular format for data, which
we produce from our augmented images and json manifest.

```
pipenv run python scripts/convert_dataset_for_tf.py
```

This will produce `./data/train.record`, `./data/test.record` which are used
by TensorFlow and `./data/validation.txt` which contains a list of images held
out for evaluation of the trained model.

Details on TFRecord files can be found
[here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/preparing_inputs.md).

#### Train Object Detection Model

Training the model can take a long time. The object detection APIs utilize
checkpoints so that it can be trained incrementally. To train the model, we
specify the (maximum) step number to train up to (100 in this example).

```
pipenv run python scripts/train_model.py 100
```

If the data was augmented with the `--all-labels` flag, it will need to be passed here as well.

TensorFlow will train from the last checkpoint (if available) up to the specified
step. Numerous files will be created in `./models/model/`, the key ones are:

* model.ckpt-100.data-00000-of-00001
* model.ckpt-100.index
* model.ckpt-100.meta

You may want to observe training using [TensorBoard](https://www.tensorflow.org/guide/summaries_and_tensorboard):

```
pipenv run tensorboard --logdir=models/model
```

#### Export Trained Model

Once the model has been trained, we can export a frozen inference graph that we
can use for prediction.

```
pipenv run python scripts/export_trained_model.py
```

The latest checkpoint will be exported, with the key output being
`./models/model/frozen_inference_graph.pb`.

Details on exporting models can be found
[here](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/exporting_models.md).

### Prediction

#### Generating Predictions

Generate a prediction for an image by passing a path to that image. Here
we use an image from our validation set.

```
pipenv run python scripts/predict.py data/images/272.png
```

A folder will be created in `./predictions` with a timestamp containing
`image.png` and `detections.json`.

#### Visualizing Predictions

To visualilze a prediction, pass the timestamp to the script.

```
pipenv run python scripts/visualize_prediction.py 1542554276
```

This will display the provided image and detected elements with
bounding boxes, class names and confidence scores indicated.

## FAQ

...

## License

This library is available as open source software under the terms of the
[MIT License](http://opensource.org/licenses/MIT).

## TODO

* confirm usage of GPU on linux
* MAYBE: multiple model pipeline, to reduce the learning required by each model (currently all in one)
* train the model on The Beast
* validate it
* Automate the ridiculous setup, at least for macOS and whatever the GPU machine runs, and make it robust.
* readme
    * document 149 data items, test, train validation split
    * finish/update setup docs
    * finish/update usage pipeline docs
    * add FAQ
* build the second half, which takes the detected objects and -> DOM

