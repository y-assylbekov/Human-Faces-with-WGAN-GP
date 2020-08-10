"""
author: Yernat M. Assylbekov
email: yernat.assylbekov@gmail.com
date: 08/10/2020
"""


import numpy as np
import glob
from PIL import Image
from skimage.transform import resize
import matplotlib.pyplot as plt

def read_preprocess_images(path, n):
    """
    reads n images from path, crops mid 128x128 sub-image and resizes them to 64x64.
    saves the images into a numpy array of shape
    (number of images, height, width, number of channels (=3 for rgb)).
    pixels of each image are normalized (in the interval [0, 1]).
    """
    images = list()
    xc, yc = 218 // 2, 178 // 2
    for file_name in glob.glob(path)[:n]:
        image = Image.open(file_name)
        image = np.asarray(image)
        image = image[xc - 64 : xc + 64, yc - 64 : yc + 64]
        image = resize(image, (64,64,3)) # resize function returns an image whose pixels have values are normalized (in [0,1])
        images.append(image)

    return np.asarray(images)

def print_save_images(images):
    """
    prints and saves into a .png file first 25 images (in 5 columns and 5 rows) from images numpy array.
    """
    fig = plt.figure(figsize=(10, 10))
    for i in range(25):
        plt.subplot(5, 5, i + 1)
        plt.imshow(images[i])
        plt.axis('off')

    fig.suptitle('Images from the training set', fontsize=16, y=0.92)
    plt.savefig('images_from_train_set.png')
    plt.show()
