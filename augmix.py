import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
# import augmentations


def apply_operation(image, operation):
    # rescale the image to the range [0, 255]
    image = np.clip(image * 255., 0, 255).astype(np.uint8)
    # Convert to PIL.Image
    img = Image.fromarray(image)
    # apply the operation
    pil_img = operation(img)
    return np.asarray(pil_img) / 255.

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


# if __name__ == "__main__":
#     image = np.array(Image.open("img/example.jpg")).astype(np.float32) / 255.
#     print(Image.open("img/example.jpg").size)
#     plt.imshow(image)
#     plt.show()
#     image_aug = augmix(image, augmentations.augmentations)
#     plt.imshow(image_aug)
#     plt.show()
