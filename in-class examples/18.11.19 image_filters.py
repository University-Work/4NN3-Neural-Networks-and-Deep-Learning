import numpy as np
import scipy
import skimage.io
import skimage.viewer


def convolve2d(image, filter):
    # Flip filter left right and up down
    filter = np.flipud(np.fliplr(filter))

    # Image output
    output = np.zeros_like(image)

    # Avoid problems of edges by zero padding the image
    # Create the image of zeroes two pixels bigger in both rows and columns
    image_padded = np.zeros((image.shape[0] + 2, image.shape[1] + 2))
    image_padded[1:-1, 1:-1] = image

    for m in range(image.shape[1]):
        for n in range(image.shape[0]):
            # element-wise multiplication of the filter and the image
            # and then add up the multiplications
            output[n, m] = (filter*image_padded[n:n+3, m:m+3]).sum()

    return output


img = skimage.io.imread('toronto-skyline.jpg', as_gray=True)
skimage.viewer.ImageViewer(img).show()

# Convolve the image with a filter that blurs the image...
blur_filter = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])/9.0
edge_filter = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])

# image_result = convolve2d(img, edge_filter)
image_result = scipy.signal.convolve2d(img, edge_filter, 'same')

skimage.viewer.ImageViewer(image_result).show()

print('DONE')
