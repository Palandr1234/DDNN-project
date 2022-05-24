import numpy as np
from PIL import Image


# CIFAR-10 constants
MEAN = [0.4914, 0.4822, 0.4465]
STD = [0.2023, 0.1994, 0.2010]


def normalize(image, mean=MEAN, std=STD):
    image = image.transpose(2, 0, 1)
    mean, std = np.array(mean), np.array(std)
    image = (image - mean[:, None, None]) / std[:, None, None]
    return image.transpose(1, 2, 0)


def apply_operation(image, operation):
    # rescale the image to the range [0, 255]
    image = np.clip(image * 255., 0, 255).astype(np.uint8)
    # Convert to PIL.Image
    img = Image.fromarray(image)
    # apply the operation
    pil_img = operation(img)
    return np.asarray(pil_img) / 255.
