from typing import List, Callable, Tuple, Union

import numpy as np
import albumentations as A
from sklearn.externals._pilutil import bytescale


class Repr(object):
    """Get string representation of object class"""

    def __repr__(self):
        return f"{self.__class__.__name__}: {self.__dict__}"


class FunctionWrapperSingle(Repr):
    """Function wrapper that returns a partial for a single input"""

    def __init__(self, function: Callable, *args, **kwargs):
        from functools import partial
        self.function = partial(function, *args, **kwargs)

    def __call__(self, inp: np.ndarray):
        return self.function(inp)


class FunctionWrapperDouble(Repr):
    """Function wrapper that returns a callable for an input-target pair"""

    def __init__(self, function: Callable, inp: bool = True, tar: bool = True, *args,
                 **kwargs):
        from functools import partial
        self.function = partial(function, *args, **kwargs)
        self.inp = inp
        self.tar = tar

    def __call__(self, inp: np.ndarray, tar: np.ndarray):
        if self.inp:
            inp = self.function(inp)
        if self.tar:
            tar = self.function(tar)
        return inp, tar


class AlbuWrapper2D(Repr):
    """Wrapper for albumentations' 2D augmentations for transform-pipeline compatibility"""

    def __init__(self, albumentation: Callable):
        self.albumentation = albumentation

    def __call__(self, inp: np.ndarray, tar: np.ndarray):
        out_dict = self.albumentation(image=inp, mask=tar)
        inp = out_dict['image']
        tar = out_dict['mask']
        return inp, tar


class Compose(object):
    """Baseclass for composing multiple transforms together"""

    def __init__(self, transforms: Union[List[Callable], Tuple[Callable]]):
        self.transforms = transforms

    def __repr__(self):
        return [str(transform) for transform in self.transforms]

    def __call__(self, *args, **kwargs):
        raise NotImplementedError()


class ComposeDouble(Compose):
    """Definition of Compose class for input-target pairs"""

    def __call__(self, inp: np.ndarray, tar: np.ndarray):
        for transform in self.transforms:
            inp, tar = transform(inp, tar)
        return inp, tar


class ComposeSingle(Compose):
    """Definition of Compose class for input-target pairs"""

    def __call__(self, inp: np.ndarray):
        for transform in self.transforms:
            inp = transform(inp)
        return inp
