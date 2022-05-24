from PIL import Image, ImageOps
import random
import numpy as np


def posterize(image):
    num_bits = random.randint(1, 4)
    return ImageOps.posterize(image, num_bits)


def rotate(image):
    degrees = random.randint(1, 30)
    if np.random.uniform() > 0.5:
        degrees = -degrees
    return image.rotate(degrees, resample=Image.BILINEAR)


def shear_x(image):
    param = np.random.uniform(0.1, 0.3)
    if np.random.uniform() > 0.5:
        param = -param
    return image.transform((image.size[0], image.size[1]),
                           Image.AFFINE, (1, param, 0, 0, 1, 0),
                           resample=Image.BILINEAR)


def shear_y(image):
    param = np.random.uniform(0.1, 0.3)
    if np.random.uniform() > 0.5:
        param = -param
    return image.transform((image.size[0], image.size[1]),
                           Image.AFFINE, (1, 0, 0, param, 1, 0),
                           resample=Image.BILINEAR)


def translate_x(image):
    param = random.randint(1, min(image.size[0], image.size[1]))
    if np.random.random() > 0.5:
        param = -param
    return image.transform((image.size[0], image.size[1]),
                           Image.AFFINE, (1, 0, param, 0, 1, 0),
                           resample=Image.BILINEAR)


def translate_y(image):
    param = random.randint(1, min(image.size[0], image.size[1]))
    if np.random.random() > 0.5:
        param = -param
    return image.transform((image.size[0], image.size[1]),
                           Image.AFFINE, (1, 0, 0, 0, 1, param),
                           resample=Image.BILINEAR)


augmentations = [ImageOps.autocontrast, ImageOps.equalize, posterize, rotate,
                 shear_x, shear_y, translate_x, translate_y]
