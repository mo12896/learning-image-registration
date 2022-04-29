"""Transforms for 3D Biomedical Tensor Images"""

from torchvision.transforms.transforms import Resize, ToPILImage, ToTensor, Normalize
import torch.nn.functional as F

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *inputs):
        for transform in self.transforms:
            if not isinstance(inputs, (list, tuple)):
                inputs = [inputs]
            inputs = transform(*inputs)
        return inputs

    def _get_transforms(self):
        return self.transforms


class Create2D(object):
    def __init__(self, axis):
        self.axis = axis

    def __call__(self, sample):
        fixed, moving = sample[0], sample[1]
        if self.axis == 'z':
            return [fixed[100,:,:], moving[100,:,:], sample[2]]
        elif self.axis == 'y':
            return [fixed[:,100,:], moving[:,100,:], sample[2]]
        elif self.axis == 'x':
            return [fixed[:,:,100], moving[:,:,100], sample[2]]
        else:
            raise AttributeError(f"Axis {self.axis} does not exist. Please choose from either x, y or z!")


class AddChannel(object):
    def __init__(self, axs=0):
        """Add channel at the chosen input axes of torch tensors
        Useful making input (H x W x D) tensor to (C x H x W x D) tensor
        Arguments
        ---------
        axs: int or sequence of ints:
            Axes at which to add channel, default 0.

        """
        self.axs = axs

    def __call__(self, sample):
        """
        Arguments
        ---------
        inputs : Tensors
            Tensors to which channels are added
        Returns
        -------
        outputs: Tensors
        """
        if not isinstance(self.axs, (tuple,list)):
            axs = [self.axs]*len(sample)
        else:
            axs = self.axs
        fix = sample[0].unsqueeze(0)
        mov = sample[1].unsqueeze(0)
        return [fix, mov, sample[2]]


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        fixed, moving = sample[0], sample[1]

        if isinstance(self.output_size, int):
            new_h, new_w = self.output_size, self.output_size
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        transform = Resize((new_h, new_w))
        #TODO: due to python version!?
        pil = ToPILImage()
        ten = ToTensor()

        fix = ten(transform(pil(fixed)))
        mov = ten(transform(pil(moving)))
        return [fix, mov, sample[2]]


# maybe not needed!
class NormalizeSample(object):
    def __init__(self, dim):
        self.dim = dim

    def __call__(self, sample):
        fixed, moving = sample[0], sample[1]

        fix = F.normalize(fixed, dim=self.dim)
        mov = F.normalize(moving, dim=self.dim)
        return [fix, mov, sample[2]]

class Standardize(object):
    pass

