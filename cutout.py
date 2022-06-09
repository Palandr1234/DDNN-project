import numpy as np


def cutout(image, num_regions, hole_height, hole_width):
    """
    Cutout data augmentation method
    :param image: image to be transformed, np.array of shape [h, w, c] if c>1 and [h,w] otherwise
    :param num_regions: number of regions to be cutout
    :param hole_height: height of each hole
    :param hole_width: width of each hole
    :return: Transformed image
    """
    height = image.shape[0]
    width = image.shape[1]

    mask = np.ones(image.shape)

    for _ in range(num_regions):
        hole_y = np.random.randint(height)
        hole_x = np.random.randint(width)

        hole_x1 = np.clip(hole_x - hole_width // 2, 0, width)
        hole_x2 = np.clip(hole_x + hole_width // 2, 0, width)
        hole_y1 = np.clip(hole_y - hole_height // 2, 0, height)
        hole_y2 = np.clip(hole_y + hole_height // 2, 0, height)

        mask[hole_y1: hole_y2, hole_x1: hole_x2, :] = 0.

    return image * mask
