"""Script to load raw EMPIRE10 data into .nii image format"""

import os
import argparse

import SimpleITK as sitk
import tempfile

import sys
sys.path.append('../')
import systemsetup as setup


image_types = ['BMPImageIO', 'BioRadImageIO', 'Bruker2dseqImageIO', 'GDCMImageIO',
               'GE4ImageIO', 'GE5ImageIO', 'GiplImageIO', 'HDF5ImageIO', 'JPEGImageIO',
               'LSMImageIO', 'MINCImageIO', 'MRCImageIO', 'MetaImageIO', 'NiftiImageIO',
               'NrrdImageIO', 'PNGImageIO', 'StimulateImageIO', 'TIFFImageIO', 'VTKImageIO']


def read_image(image_path, image_type, verbose=False):
    if image_type not in image_types:
        raise AttributeError("Image type must be a string and supported by the SimpleITK API!")
    if '.' not in image_path:
        raise IOError("Image path must contain a valid image format! See docs.")
    file_reader = sitk.ImageFileReader()
    file_reader.SetImageIO(image_type)
    file_reader.SetFileName(image_path)
    try:
        image = file_reader.Execute()
        print('Read image: ' + image_path)
        if verbose:
            print(file_reader)
        return image
    except Exception as err:
        print('Reading failed: ', err)


def read_raw(binary_file_name, image_size, sitk_pixel_type, image_spacing=None,
             image_origin=None, big_endian=False):
    """
    From https://simpleitk.readthedocs.io/en/master/link_RawImageReading_docs.html
    Read a raw binary scalar image.

    Parameters
    ----------
    binary_file_name (str): Raw, binary image file content.
    image_size (tuple like): Size of image (e.g. [2048,2048])
    sitk_pixel_type (SimpleITK pixel type: Pixel type of data (e.g.
        sitk.sitkUInt16).
    image_spacing (tuple like): Optional image spacing, if none given assumed
        to be [1]*dim.
    image_origin (tuple like): Optional image origin, if none given assumed to
        be [0]*dim.
    big_endian (bool): Optional byte order indicator, if True big endian, else
        little endian.

    Returns
    -------
    SimpleITK image or None if fails.
    """

    pixel_dict = {sitk.sitkUInt8: 'MET_UCHAR',
                  sitk.sitkInt8: 'MET_CHAR',
                  sitk.sitkUInt16: 'MET_USHORT',
                  sitk.sitkInt16: 'MET_SHORT',
                  sitk.sitkUInt32: 'MET_UINT',
                  sitk.sitkInt32: 'MET_INT',
                  sitk.sitkUInt64: 'MET_ULONG_LONG',
                  sitk.sitkInt64: 'MET_LONG_LONG',
                  sitk.sitkFloat32: 'MET_FLOAT',
                  sitk.sitkFloat64: 'MET_DOUBLE'}
    direction_cosine = ['1 0 0 1', '1 0 0 0 1 0 0 0 1',
                        '1 0 0 0 0 1 0 0 0 0 1 0 0 0 0 1']
    dim = len(image_size)
    header = ['ObjectType = Image\n'.encode(),
              (f'NDims = {dim}\n').encode(),
              ('DimSize = ' + ' '.join([str(v) for v in image_size]) + '\n')
              .encode(),
              ('ElementSpacing = ' + (' '.join([str(v) for v in image_spacing])
                                      if image_spacing else ' '.join(
                  ['1'] * dim)) + '\n').encode(),
              ('Offset = ' + (
                  ' '.join([str(v) for v in image_origin]) if image_origin
                  else ' '.join(['0'] * dim) + '\n')).encode(),
              ('TransformMatrix = ' + direction_cosine[dim - 2] + '\n')
              .encode(),
              ('ElementType = ' + pixel_dict[sitk_pixel_type] + '\n').encode(),
              'BinaryData = True\n'.encode(),
              ('BinaryDataByteOrderMSB = ' + str(big_endian) + '\n').encode(),
              # ElementDataFile must be the last entry in the header
              ('ElementDataFile = ' + os.path.abspath(
                  binary_file_name) + '\n').encode()]
    fp = tempfile.NamedTemporaryFile(suffix='.mhd', delete=False)

    print(header)

    # Not using the tempfile with a context manager and auto-delete
    # because on windows we can't open the file a second time for ReadImage.
    fp.writelines(header)
    fp.close()
    img = sitk.ReadImage(fp.name)
    os.remove(fp.name)
    return img


def convert_raw_to_img(raw_folder, img_folder, pixel_type, image_format, verbose=False):
    output_dir = setup.DATA_DIR + img_folder
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)
        print(f"Created directory: {output_dir}")

    input_dir = setup.DATA_DIR + raw_folder
    ids = list(set([x.split('_')[0] for x in os.listdir(input_dir)]))
    modes = ['_Fixed', '_Moving']

    for id in ids:
        for mode in modes:
            # Read Meta data
            image = id + mode
            image_type = 'MetaImageIO'
            raw_image = input_dir + image + '.mhd'
            meta_image = read_image(raw_image, image_type, verbose=verbose)

            # Read Image
            big_endian = False
            sitk_pixel_type = pixel_type
            binary_file_name = input_dir + image + '.raw'
            image_size = meta_image.GetSize()
            image_spacing = meta_image.GetSpacing()
            image_origin = meta_image.GetOrigin()

            img = read_raw(binary_file_name, image_size, sitk_pixel_type, image_spacing=image_spacing,
                           image_origin=image_origin, big_endian=big_endian)

            # Safe Image in new format
            file_out = output_dir + image + image_format
            if not os.path.isfile(file_out):
                if image_format == '.jpg':
                    nda = sitk.GetArrayFromImage(img)
                    sitk.WriteImage(nda[0, :], file_out)
                else:
                    sitk.WriteImage(img, file_out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load EMPIRE10 data into .nii image format" ')
    parser.add_argument('--scans', dest='convert_scans', type=bool, default=False,
                        help='Load the scan images.')
    parser.add_argument('--masks', dest='convert_masks', type=bool, default=False,
                        help='Load the mask images.')
    args = parser.parse_args()

    if args.convert_scans:
        scans_raw_folder = 'raw/EMPIRE10/scans/'
        scans_img_folder = 'interim/EMPIRE10/scans/'
        scans_pixel_type = sitk.sitkInt16
        scans_image_format = '.nii'

        convert_raw_to_img(scans_raw_folder, scans_img_folder, scans_pixel_type,
                           scans_image_format, verbose=False)

    if args.convert_masks:
        masks_raw_folder = 'raw/EMPIRE10/masks/'
        masks_img_folder = 'interim/EMPIRE10/masks/'
        masks_pixel_type = sitk.sitkUInt8
        masks_image_format = '.nii'

        convert_raw_to_img(masks_raw_folder, masks_img_folder, masks_pixel_type,
                           masks_image_format, verbose=False)