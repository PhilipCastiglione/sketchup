import os

import imgaug
from imgaug import augmenters

from src.dataset_image_region import DatasetImageRegion
from src.dataset_image import DatasetImage

class Augmenter:
    def __init__(self, original_dataset):
        self.dataset = original_dataset
        self.new_images = []
        self.augmentation_sequence = self._build_augmentation_sequence()

    def augment(self):
        self.new_images = self._generate_augmented_images()
        self.dataset.images += self.new_images

    def write(self):
        for image in self.new_images:
            image.write()
        self.dataset.write()

    def _build_augmentation_sequence(self):
        # to make this deterministic for reproducibility, we could provide a known seed here:
        # imgaug.seed(1)
        at_p_25 = lambda aug: augmenters.Sometimes(0.25, aug)
        at_p_50 = lambda aug: augmenters.Sometimes(0.5, aug)
        at_p_100 = lambda aug: aug

        scale = at_p_50(augmenters.Affine(scale={"x": (0.5, 1.25), "y": (0.5, 1.25)}))
        crop_pad = at_p_50(augmenters.CropAndPad(percent=(-0.1,0.1), pad_mode=imgaug.ALL))
        blur = at_p_25(augmenters.GaussianBlur(sigma=(0.0, 0.30)))
        salt_pepper = at_p_25(augmenters.SaltAndPepper(0.1))
        rotate = at_p_100(augmenters.Affine(rotate=(-5, 5)))

        seq = augmenters.Sequential([scale, crop_pad, blur, salt_pepper, rotate], random_order=True)
        return seq.to_deterministic()

    def _generate_augmented_images(self):
        def new_image(old_image, new_image_array, new_regions):
            new_id = old_image.guid + "AUGMENTED"
            return DatasetImage(new_id, old_image.width, old_image.height, new_regions, new_image_array)

        originals = self.dataset.images
        arrays = self._augmented_image_arrays(originals)
        regions = self._augmented_regions(originals)

        augmented_images = [new_image(original, ia, r) for original, ia, r in zip(originals, arrays, regions)]

        return augmented_images

    def _augmented_image_arrays(self, originals):
        image_arrays = [image.array for image in originals]
        return self.augmentation_sequence.augment_images(image_arrays)

    def _augmented_regions(self, originals):
        def bounding_boxes_on_image(image):
            return imgaug.BoundingBoxesOnImage(self._regions_to_bounding_boxes(image.regions), shape=(image.height, image.width, 3))

        image_bounding_boxes = [bounding_boxes_on_image(image) for image in originals]

        augmented_image_bounding_boxes = self.augmentation_sequence.augment_bounding_boxes(image_bounding_boxes)
        trimmed_augmented_image_bounding_boxes = [ibb.remove_out_of_image().cut_out_of_image() for ibb in augmented_image_bounding_boxes]

        augmented_regions = [self._bounding_boxes_to_regions(ibb.bounding_boxes) for ibb in trimmed_augmented_image_bounding_boxes]

        return augmented_regions

    def _regions_to_bounding_boxes(self, regions):
        to_bounding_box = lambda r: imgaug.BoundingBox(x1=r.x1, y1=r.y1, x2=r.x2, y2=r.y2, label=r.label)
        return [to_bounding_box(r) for r in regions]

    def _bounding_boxes_to_regions(self, bounding_boxes):
        to_region = lambda bb: DatasetImageRegion(bb.x1, bb.y1, bb.x2, bb.y2, bb.label)
        return [to_region(bb) for bb in bounding_boxes]

