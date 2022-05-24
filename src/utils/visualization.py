import SimpleITK as sitk
import matplotlib.pyplot as plt

import numpy as np


def showITKImage(img):
    """
    Plot ITK Image
    :param img: img
    :type img: SimpleITK.SimpleITK.Image
    :return: plt-plot
    :rtype: numpy array
    """
    # change point of view through index slicing
    aimg = sitk.GetArrayViewFromImage(img)[:, 100, :]
    plt.imshow(aimg)


def plot_images_in_row(images: list = None):
    fig = plt.figure()
    row_len = len(images)
    for i in range(row_len):
        fig.add_subplot(1, row_len, i + 1)
        plt.imshow(images[i].view(images[i].shape[1], images[i].shape[2], 1).cpu())
    plt.show()
