import torch
import numpy as np
from skorch.utils import to_numpy, to_tensor, to_device

# ---------------------------------------------------------------------------
# mask_classes
# ---------------------------------------------------------------------------

def mask_classes(image: np.ndarray|torch.Tensor) -> np.ndarray|torch.Tensor:
    """
    Retrieve a black/white image (int in8 or float) and convert it in
    a image with multiple channels, one for each class

    if image in int8 -> each int value is a class
    if image in float -> convert it in int8

    :param image:
    :return:
    """
    if isinstance(image, torch.Tensor):
        return _mask_classes_tensor(image)
    else:
        return _mask_classes_array(image)


def _mask_classes_tensor(image: torch.Tensor) -> torch.Tensor:
    image = to_numpy(image)
    imclasses = _mask_classes_array(image)
    imclasses = to_tensor(imclasses , "cpu")
    return imclasses


def _mask_classes_array(image: np.ndarray) -> np.ndarray:
    if image.dtype in [float, np.float16, np.float32, np.float64]:
        image = (image*255).astype(np.int8)

    min_class = image.min()
    max_class = image.max()
    num_classes = max_class - min_class + 1
    imshape = image.shape + tuple([num_classes])

    imclasses = np.zeros(imshape, dtype=float)

    if len(imclasses.shape) == 3:
        for iclass in range(num_classes):
            imclasses[:,:,iclass][image == (min_class + iclass)] = 1.
        imclasses = np.swapaxes(imclasses, 0, 2)
    elif len(imclasses.shape) == 4:
        for iclass in range(num_classes):
            imclasses[:,:,:,iclass][image == (min_class + iclass)] = 1.
        imclasses = np.swapaxes(imclasses, 1, 3)
    else:
        raise ValueError("Unsupported shape")

    # move classes dimension in 2nd position

    return imclasses

# ---------------------------------------------------------------------------
# compose_classes
# ---------------------------------------------------------------------------


