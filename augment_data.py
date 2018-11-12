"""
# PURPOSE

Our image data can be augmented to improve our model:

    https://medium.com/nanonets/how-to-use-deep-learning-when-you-have-limited-data-part-2-data-augmentation-c26971dc8ced

We will use this library:

https://github.com/aleju/imgaug

The following transformations are performed, each at a rate r per image:

- r = 0.5; scaling (0.5 >= x*y >= 1.25)
- r = 0.5; crop + pad (crop and/or pad x,y -10% >= xy >= 10%)
- r = 0.25; gaussian blur (0 >= sig >= 3.0)
- r = 0.25; salt and pepper (p = 0.1)
- r = 1.0; affine rotate (-10 >= deg >= 10)

# USAGE

pipenv run python augment_data

# NOTES

Run over 150 images, another 150 images will be generated. This will take a few minutes.

If generated images already exist they will be overwritten.

WARNING: Generation is based on a random seed, so regenerated data sets will not
be identical.
"""

import os
import paths
import imgaug
from imgaug import augmenters as iaa
import cv2
from PIL import Image
from datetime import datetime
import json

def build_augmentation_sequence():
    at_p_25 = lambda aug: iaa.Sometimes(0.25, aug)
    at_p_50 = lambda aug: iaa.Sometimes(0.5, aug)
    at_p_100 = lambda aug: aug

    scale = at_p_50(iaa.Affine(scale={"x": (0.5, 1.25), "y": (0.5, 1.25)}))
    crop_pad = at_p_50(iaa.CropAndPad(percent=(-0.1,0.1), pad_mode=imgaug.ALL))
    blur = at_p_25(iaa.GaussianBlur(sigma=(0.0, 0.30)))
    salt_pepper = at_p_25(iaa.SaltAndPepper(0.1))
    rotate = at_p_100(iaa.Affine(rotate=(-10, 10)))

    seq = iaa.Sequential([scale, crop_pad, blur, salt_pepper, rotate], random_order=True)
    return seq.to_deterministic()

def load_image_array(datum):
    return cv2.imread(paths.IMAGES + datum["id"] + ".png")

def region_to_bounding_box(datum, region):
    x1 = datum["width"] * region["left"]
    y1 = datum["height"] * region["top"]
    x2 = x1 + datum["width"] * region["width"]
    y2 = y1 + datum["height"] * region["height"]
    label = region["tagName"]

    return imgaug.BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2, label=label)

def bounding_box_to_region(datum, bounding_box):
    return {
        "tagName": bounding_box.label,
        "left": bounding_box.x1 / datum["width"],
        "top": bounding_box.y1 / datum["height"],
        "width": (bounding_box.x2 - bounding_box.x1) / datum["width"],
        "height": (bounding_box.y2 - bounding_box.y1) / datum["height"]
    }

def load_image_bounding_boxes(datum):
    bounding_boxes = [region_to_bounding_box(datum, r) for r in datum["regions"]]
    return imgaug.BoundingBoxesOnImage(bounding_boxes, shape=(datum["height"], datum["width"], 3))

def generate_augmented_images(dataset, augmenter):
    images = [load_image_array(d) for d in dataset]
    return augmenter.augment_images(images)

def generate_augmented_bounding_boxes(dataset, augmenter):
    images_bounding_boxes = [load_image_bounding_boxes(d) for d in dataset]
    augmented_bounding_boxes = augmenter.augment_bounding_boxes(images_bounding_boxes)
    return [bb.remove_out_of_image().cut_out_of_image() for bb in augmented_bounding_boxes]

def generate_augmented_datum(datum, image, bounding_boxes):
    return {
        "id": datum["id"] + "AUGMENTED",
        "width": datum["width"],
        "height": datum["height"],
        "regions": [bounding_box_to_region(datum, bb) for bb in bounding_boxes]
    }

def generate_augmented_data(dataset, images, images_bounding_boxes):
    return [generate_augmented_datum(d, images[i], images_bounding_boxes[i].bounding_boxes) for i, d in enumerate(dataset)]

def write_new_data(dataset, augmented_data, new_images):
    for idx, datum in enumerate(augmented_data):
        path = paths.IMAGES + datum["id"] + ".png"
        with open(path, 'wb') as f:
            Image.fromarray(new_images[idx]).save(f, format="PNG", optimize=True)

    with open(paths.INPUT_DATASET, 'w') as f:
        json.dump(dataset + augmented_data, f, indent=2)

def augment_data(dataset):
    # to make this deterministic for reproducibility, we could provide a known seed here:
    # imgaug.seed(1)
    augmenter = build_augmentation_sequence()

    augmented_images = generate_augmented_images(dataset, augmenter)
    augmented_bounding_boxes = generate_augmented_bounding_boxes(dataset, augmenter)
    augmented_data = generate_augmented_data(dataset, augmented_images, augmented_bounding_boxes)

    write_new_data(dataset, augmented_data, augmented_images)


if __name__ == "__main__":
    print("Augmentation beginning: " + datetime.now().isoformat())

    with open(paths.ORIGINAL_DATASET, 'r') as f:
        dataset = json.load(f)

    augment_data(dataset)

    print("Augmentation complete: " + datetime.now().isoformat())
