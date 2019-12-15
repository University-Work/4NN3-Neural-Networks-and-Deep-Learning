import numpy as np
import scipy
import skimage.io
from skimage import io, viewer
import random


def convolve2d(image, kernel):
    '''
    This function takes an image and a kernel
    and returns the convolution of them \n

    Args:\n
    \t  image: a numpy array of size [image_height, image_width]
    \t  kernel: a numpy array of size [kernel_height, kernel_width]
    Returns:\n
    \t  a numpy array of size [image_height, image_width] ( convolution output )
    '''

    kernel = np.flipud(np.fliplr(kernel))  # Flip kernel left right and up down
    output = np.zeros_like(image)  # convolution output

    # Avoid problems of edges by zero padding the image
    # Create the image of zeroes two pixels bigger in both rows and columns
    image_padded = np.zeros((image.shape[0] + 2, image.shape[1] + 2))
    image_padded[1:-1, 1:-1] = image

    for m in range(image.shape[1]):
        for n in range(image.shape[0]):
            # element-wise multiplication of the kernel and the image
            # and then add up the multiplications
            output[n, m] = (kernel*image_padded[n:n+3, m:m+3]).sum()

    return output


img = io.imread('toronto-skyline.jpg', as_gray=True)  # load image as grayscale
print('image matrix size: ', img.shape)  # print the size of the image
print('\n First 5 columns and rows of the image matrix: \n', img[:5, :5]*255)
viewer.ImageViewer(img).show()  # plot the image


# Convolve the image with a filter that blurs the image...
blur_filter = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])/9.0
edge_filter = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
sharpen_filter = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])

random.seed()
rand_num = random.random()
random_filter = np.array([[7, 9, 23],
                          [76, 91, 7],
                          [64, 90, 32]])/255

# image_sharpen = convolve2d(img, sharpen_filter);
image_sharpen = scipy.signal.convolve2d(img, random_filter, 'same')
print('\n First 5 columns and rows of the image_sharpen matrix: \n',
      image_sharpen[:5, :5]*255)

viewer.ImageViewer(image_sharpen).show()
print('DONE')
