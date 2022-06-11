import torch
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt


def mixup(inputs, outputs=None, device="cpu", lam=0.5):
    """
    Mixup data augmentation method
    :param inputs: batch of input images, torch.tensor
    :param outputs: batch of output images if any, torch.tensor
    :param device: device, torch.device or str
    :param lam: mixup strength
    :return: mixed input and output tensors
    """

    batch_size = inputs.size(0)
    # get tensor permutation
    index = torch.randperm(batch_size).to(device)

    # mix input images
    inputs_mixed = lam * inputs + (1 - lam) * inputs[index, :]

    # mix output images if there are any
    if outputs is not None:
        outputs_mixed = lam * outputs + (1 - lam) * outputs[index, :]
    else:
        outputs_mixed = None

    return inputs_mixed, outputs_mixed


if __name__ == "__main__":
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((225, 225))
    ])
    images = [Image.open("img/example.jpg"), Image.open("img/example2.jpg"),
              Image.open("img/example3.jpg")]
    for i, img in enumerate(images):
        images[i] = transform(images[i])

    images = torch.stack(images).float() / 255.
    print(images.shape)
    image_aug, _ = mixup(images)
    for img in image_aug:
        plt.imshow(img.permute(1, 2, 0)*255.)
        plt.show()
