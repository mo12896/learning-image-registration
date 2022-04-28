
import SimpleITK as sitk
import matplotlib.pyplot as plt


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