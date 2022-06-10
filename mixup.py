import numpy as np
import torch


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
