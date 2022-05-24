import numpy as np
from utils import apply_operation
from PIL import Image
import matplotlib.pyplot as plt
import augmentations


def augmix(image, augmentations, alpha=1., k=3, depth=3):
    """
    Augment and mix technique
    :param image: image to be transformed
    :param augmentations:
    :param alpha: coefficient for Beta and Dirichlet distributions
    :param k: width of augmentation chain
    :param depth: depth of augmentation chain
    :return: Transformed image
    """
    # Fill augmented image with zeros
    result = np.zeros_like(image).astype(np.float32)
    # Sample mixing weights
    w = np.random.dirichlet([alpha] * k).astype(np.float32)
    for i in range(k):
        image_aug = image.copy()
        for _ in range(depth):
            # sample operation
            operation = np.random.choice(augmentations)
            # apply the operation
            image_aug = apply_operation(image_aug, operation)
        result += w[i] * image_aug
    m = np.float32(np.random.beta(alpha, alpha))
    # Interpolate the final result
    result = m * image + (1 - m) * result
    return result