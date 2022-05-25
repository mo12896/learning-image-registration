from typing import List, Callable, Tuple

import numpy as np
import albumentations as A
from sklearn.externals._pilutil import bytescale
from skimage.util import crop


class RandomVerticalFlip(object):

    def __init__(self, prob: float = 0.5):
        assert 0 <= prob <= 1, "Probability value must lie between 0 and 1!"
        self.prob = prob
        self.flip_axis = self.get_flip_axis()

    def get_flip_axis(self):
        return np.random.binomial(1, self.prob)

    def __call__(self, inp: np.ndarray):
        if self.flip_axis:
            # TODO: there is a problem with the flip direction!
            inp_flipped = np.flip(inp, axis=self.flip_axis).copy()
        else:
            return inp
        return inp_flipped


class RandomCrop():
    pass


class RandomBrightnessContrast():
    pass


class RandomRotate():
    pass


class RandomNoise():
    pass


class RandomScaling():
    pass
