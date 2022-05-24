import numpy as np
from PIL import Image


def normalize(image, mean, std):
    image = image.transpose(2, 0, 1)
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
