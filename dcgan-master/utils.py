"""
utils.py这个文件是为了数据增强使用的
因为在paper中我们需要数据量有限,需要使用数据增强进行数据量的扩展的
"""

from random import shuffle
import scipy.misc
import numpy as np


"""
center_crop
args:
    x: 需要进行进行裁剪的图像
    crop_h: 裁剪图像大小宽度
    crop_w: 裁剪图像大小
    resize_w: resize the image size, often we set the resize_w is same as the crop_h
"""


def center_crop(x, crop_h, crop_w=None, resize_w=64):
    if crop_w is None:
        crop_w = crop_h   # if dault we set the crop_w = crop_h
    h, w = x.shape[:2]  # the original image shape in which h = x.shape[0] and w = x.shape[1]
    j = int(round((h - crop_h)/2.))  # Question? here, why do we need to have a round function?
    i = int(round((w - crop_w)/2.))
    return scipy.misc.imresize(x[j:j+crop_h, i:i+crop_w],
                               [resize_w, resize_w])  # the first parameter is the image that we want to resize, and the second is the resize image

# def merge(images, size):
#     h, w = images.shape[1], images.shape[2]
#     img = np.zeros((h * size[0], w * size[1], 3))
#     for idx, image in enumerate(images):
#         i = idx % size[1]
#         j = idx // size[1]
#         img[j*h:j*h+h, i*w:i*w+w, :] = image
#     return img


"""
transform:
args:
    image: the origin image we want to transform
    npx: the resize image size
    is_crop: whether we need to crop the image. Ture: crop, False: not crop
    resize_w: we want to resize of image size    
"""
def transform(image, npx=64, is_crop=True, resize_w=64):
    if is_crop:  # when crop
        cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:  # we don not use crop
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.  # scale the origin the image to [-1, 1], if /255 we scale the image to [0,1]

# def inverse_transform(images):
#     return (images+1.)/2.


"""
imread 
args:
    path: the image path you want to read 
    is_grayscale, if True prove is a RGB image and we need to set flatten as True, else is False prove is a gray mode 
    Attention: the scipy.misc.imread return the narray so need to convert it to the float, because we need put it as a
     input image, so the type of the image must be a float instead of a ndarray 
"""
def imread(path, is_grayscale = False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten=True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

# def imsave(images, size, path):
#     return scipy.misc.imsave(path, merge(images, size))


"""
get_image:
    image_path: the image path we want to read 
    image_size : after crop the image size 
    is_crop: whether need we to crop the image 
    resize_w: after crop we want to resize the image size 
    is_grayscale: the input image is gray image or RGB image and we need to handle it differently 
    return : return the crop(optional) gray(optional) resize image!
"""
def get_image(image_path, image_size, is_crop=True, resize_w=64, is_grayscale = False):
    return transform(imread(image_path, is_grayscale), image_size, is_crop, resize_w)

# def save_images(images, size, image_path):
#     return imsave(inverse_transform(images), size, image_path)
